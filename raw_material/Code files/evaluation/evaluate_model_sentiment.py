import os
import sys
import argparse
import csv
import numpy as np
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from data_utils import load_sentiment_splits

BATCH_SIZE = 32
MAX_LENGTH = 256
PERCENTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": p_m,
        "recall_macro": r_m,
        "f1_macro": f1_m,
        "f1_weighted": f1_w,
    }


def evaluate_single_model(model_path, tokenized_test):
    print(f"\nEvaluating: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"  [Skipping] Could not load model from {model_path}")
        return None

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/eval", report_to="none", do_train=False,
                               per_device_eval_batch_size=BATCH_SIZE),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer.evaluate(tokenized_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]

    # Load only the held-out test split (never seen during training or checkpoint selection)
    _, _, test_df = load_sentiment_splits(cfg, config.DATASET_ROOT)
    print(f"Loaded {len(test_df)} held-out test samples.")

    # Tokenize once (model-agnostic since all variants share the same base tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])
    tokenized_test = test_ds.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["text"],
    )

    # Prepare CSV
    output_folder = "./Results/scores"
    os.makedirs(output_folder, exist_ok=True)
    csv_file = os.path.join(output_folder, f"evaluation_summary_{args.lang}.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Type", "Percent", "Accuracy", "F1_Macro", "F1_Weighted", "Precision", "Recall"])

    # Models to evaluate
    models_to_check = [
        {"type": "baseline", "percent": 0, "path": f"./output/{args.lang}-baseline"},
    ]
    for p in PERCENTS:
        models_to_check.append({"type": "neft", "percent": p, "path": f"./output/{args.lang}-neft/{p}percent"})
    for p in PERCENTS:
        models_to_check.append({"type": "probeless", "percent": p, "path": f"./output/{args.lang}-probeless/{p}percent"})

    for item in models_to_check:
        res = evaluate_single_model(item["path"], tokenized_test)
        if res:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    item["type"], item["percent"],
                    f"{res['eval_accuracy']:.4f}", f"{res['eval_f1_macro']:.4f}",
                    f"{res['eval_f1_weighted']:.4f}",
                    f"{res['eval_precision_macro']:.4f}", f"{res['eval_recall_macro']:.4f}",
                ])
            print(f"  -> Acc: {res['eval_accuracy']:.4f} | F1_weighted: {res['eval_f1_weighted']:.4f} | F1_macro: {res['eval_f1_macro']:.4f}")

    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    main()