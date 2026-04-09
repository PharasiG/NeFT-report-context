import os
import sys
import argparse
import torch
import numpy as np
import datasets
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer,
    TrainingArguments, TrainerCallback, EarlyStoppingCallback,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from data_utils import load_sentiment_splits


# --- NeFT Callback (identical to POS/NER) ---
class NeFTCallback(TrainerCallback):
    def __init__(self, mask_dict):
        super().__init__()
        self.mask_dict = mask_dict

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if "output.dense.weight" in name:
                for layer_idx in range(12):
                    if f"layer.{layer_idx}.output.dense.weight" in name:
                        mask_key = f"{layer_idx}_out"
                        if mask_key in self.mask_dict:
                            mask = self.mask_dict[mask_key].to(param.device)
                            mask = mask.view(768, 3072)
                            param.grad *= mask
                            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]
    model_name = cfg["model_name"]
    mask_dir = os.path.join(config.OUTPUT_MASK_DIR, args.lang)

    train_df, val_df, _ = load_sentiment_splits(cfg, config.DATASET_ROOT)
    
    unique_labels = [int(x) for x in sorted(train_df["label"].unique())]
    num_labels = len(unique_labels)
    label2id = {str(label): idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: str(label) for idx, label in enumerate(unique_labels)}

    print(f"Starting Probeless Training for: {args.lang}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df[["text", "label"]]),
        "validation": datasets.Dataset.from_pandas(val_df[["text", "label"]]),
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    percents = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for p in percents:
        print(f"\n===== {args.lang} | {p}% probeless mask =====")

        mask_path = os.path.join(mask_dir, f"{p}probeless_mask.pt")
        if not os.path.exists(mask_path):
            print(f"Error: Mask not found at {mask_path}.")
            continue

        mask_dict = torch.load(mask_path)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, id2label=id2label, label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=f"./output/{args.lang}-probeless/{p}percent",
            num_train_epochs=20,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=200,
            report_to="none",
            save_total_limit=3,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[NeFTCallback(mask_dict), EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()
        trainer.save_model(f"./output/{args.lang}-probeless/{p}percent")

        del trainer, model, mask_dict
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
