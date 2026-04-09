import os
import torch
import datasets
import argparse
import sys
import numpy as np
import evaluate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, Trainer,
    TrainingArguments, DataCollatorForTokenClassification, TrainerCallback,
    EarlyStoppingCallback
)

# --- NeFT Callback ---
class NeFTCallback(TrainerCallback):
    def __init__(self, mask_dict):
        super().__init__()
        self.mask_dict = mask_dict

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get('model', None)
        if model is None:
            return

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if 'output.dense.weight' in name:
                for layer_idx in range(12):
                    if f'layer.{layer_idx}.output.dense.weight' in name:
                        mask_key = f"{layer_idx}_out"
                        if mask_key in self.mask_dict:
                            mask = self.mask_dict[mask_key].to(param.device)
                            mask = mask.view(768, 3072)
                            param.grad *= mask
                            break

# --- Helper  ---
def read_conllu(file_path):
    sentences, labels = [], []
    tokens, tags = [], []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue
            parts = line.split()
            if "-" in parts[0] or "." in parts[0]:
                continue
            tokens.append(parts[1])
            tags.append(parts[3])
    return sentences, labels

# --- Main Training Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py")
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]
    model_name = cfg["model_name"]
    mask_dir = os.path.join(config.OUTPUT_MASK_DIR, args.lang)
    data_dir = os.path.join(config.DATASET_ROOT, cfg["dataset_folder"])

    train_file = os.path.join(data_dir, f"{cfg['conllu_prefix']}-ud-train.conllu")
    dev_file = os.path.join(data_dir, f"{cfg['conllu_prefix']}-ud-dev.conllu")

    print(f"Starting Training for: {args.lang}")
    print(f"Using model: {model_name}")
    print(f"Data Source: {train_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_tokens, train_tags = read_conllu(train_file)
    dev_tokens, dev_tags = read_conllu(dev_file)

    unique_tags = sorted({tag for sent in train_tags for tag in sent})
    label2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2label = {i: tag for tag, i in label2id.items()}

    metric = evaluate.load("seqeval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"tokens": train_tokens, "labels": train_tags}),
        "validation": datasets.Dataset.from_dict({"tokens": dev_tokens, "labels": dev_tags}),
    })

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=256
        )
        aligned_labels = []
        for batch_idx in range(len(tokenized["input_ids"])):
            word_ids = tokenized.word_ids(batch_index=batch_idx)
            label_ids = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word_idx:
                    label_ids.append(label2id[examples["labels"][batch_idx][word_idx]])
                else:
                    label_ids.append(-100)
                prev_word_idx = word_idx
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    percents = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    for p in percents:
        print(f"\n===== {args.lang} | {p}% probeless mask =====")

        mask_path = os.path.join(mask_dir, f"{p}probeless_mask.pt")
        if not os.path.exists(mask_path):
            print(f"Error: Mask not found at {mask_path}.")
            continue

        mask_dict = torch.load(mask_path)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=f"./output/{args.lang}-probeless/{p}percent",
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=500,
            report_to="none",
            save_total_limit=3
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[NeFTCallback(mask_dict), EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train()
        trainer.save_model(f"./output/{args.lang}-probeless/{p}percent")

        del trainer, model, mask_dict
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()