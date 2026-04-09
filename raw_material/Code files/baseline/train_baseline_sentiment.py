import os
import sys
import argparse
import numpy as np
import datasets
import transformers
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from data_utils import load_sentiment_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py")
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]
    model_name = cfg["model_name"]

    print(f"Start Sentiment Baseline Training for: {args.lang}")
    print(f"Using model: {model_name}")
    transformers.set_seed(42)

    train_df, val_df, test_df = load_sentiment_splits(cfg, config.DATASET_ROOT)
    
    unique_labels = [int(x) for x in sorted(train_df["label"].unique())]
    num_labels = len(unique_labels)
    label2id = {str(label): idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: str(label) for idx, label in enumerate(unique_labels)}
    
    print(f"Num labels: {num_labels} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} (held out)")

    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train_df[["text", "label"]]),
        "validation": datasets.Dataset.from_pandas(val_df[["text", "label"]]),
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    output_dir = f"./output/{args.lang}-baseline"

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model()
    print(f"Baseline model saved to {output_dir}")


if __name__ == "__main__":
    main()
