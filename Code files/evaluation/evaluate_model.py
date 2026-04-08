import os
import sys
import argparse
import csv
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # Import config.py

# ===== CONSTANTS =====
BATCH_SIZE = 16
MAX_LENGTH = 256
PERCENTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def load_ud_test_data(test_file_path):
    """
    Reads the CoNLL-U test file and returns a raw Dataset object.
    """
    print(f"Loading test data from: {test_file_path}")
    sentences_tokens, sentences_tags = [], []
    cur_toks, cur_tags = [], []

    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_toks:
                    sentences_tokens.append(cur_toks)
                    sentences_tags.append(cur_tags)
                    cur_toks, cur_tags = [], []
                continue
            if line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) != 10:
                continue

            tok_id = parts[0]
            if "-" in tok_id or "." in tok_id:
                continue

            form = parts[1]
            upos = parts[3]
            cur_toks.append(form)
            cur_tags.append(upos)

    if cur_toks:
        sentences_tokens.append(cur_toks)
        sentences_tags.append(cur_tags)

    return Dataset.from_dict({"tokens": sentences_tokens, "tags": sentences_tags})

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    all_labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized.word_ids(batch_index=i)
        word_labels = examples["tags"][i]
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                # If the test set has a tag unknown to the model (rare), ignore it or map to something else.
                # Here we default to -100 (ignore) if not found, to prevent crash.
                tag = word_labels[word_id]
                label_ids.append(label2id.get(tag, -100))
            else:
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true, y_pred = [], []
    for pred_row, label_row in zip(preds, labels):
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            y_true.append(int(l))
            y_pred.append(int(p))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0],
        "recall_macro": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1],
        "f1_macro": precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2],
    }

def evaluate_single_model(model_path, raw_test_dataset):
    print(f"\nEvaluating: {model_path}")
    
    # 1. Load Model & Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    except OSError:
        print(f"  [Skipping] Could not load model from {model_path}")
        return None

    # 2. Get Label Map directly from the trained model config
    label2id = model.config.label2id
    id2label = model.config.id2label

    # 3. Process Dataset
    tokenized_test = raw_test_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True,
        remove_columns=raw_test_dataset.column_names,
        desc="Tokenizing"
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/eval", report_to="none", do_train=False),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 5. Evaluate
    results = trainer.evaluate(tokenized_test)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py")
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]
    
    # 1. Define Test File Path
    dataset_dir = os.path.join(config.DATASET_ROOT, cfg["dataset_folder"])
    test_file = os.path.join(dataset_dir, f"{cfg['conllu_prefix']}-ud-test.conllu")
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # 2. Load Raw Test Data (Once)
    raw_test_ds = load_ud_test_data(test_file)
    print(f"Loaded {len(raw_test_ds)} test sentences.")

    # 3. Prepare CSV Output
    output_folder = "./evaluation/scores"
    os.makedirs(output_folder, exist_ok=True)  # Creates the folder if it doesn't exist
    
    csv_file = os.path.join(output_folder, f"evaluation_summary_{args.lang}.csv")
    
    print(f"Results will be saved to: {csv_file}")

    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model_Type", "Percent", "Accuracy", "F1_Macro", "Precision", "Recall"])

    # 4. Define Models to Evaluate
    models_to_check = []
    
    # Add Baseline
    models_to_check.append({
        "type": "baseline",
        "percent": 0,
        "path": f"./output/{args.lang}-baseline"
    })

    # Add NeFT (5% - 100%)
    for p in PERCENTS:
        models_to_check.append({
            "type": "neft",
            "percent": p,
            "path": f"./output/{args.lang}-neft/{p}percent"
        })
    
    # Add Probeless (5% - 100%)
    for p in PERCENTS:
        models_to_check.append({
            "type": "probeless",
            "percent": p,
            "path": f"./output/{args.lang}-probeless/{p}percent"
        })

    # 5. Loop and Evaluate
    for item in models_to_check:
        res = evaluate_single_model(item["path"], raw_test_ds)
        
        if res:
            # Write to CSV
            with open(csv_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    item["type"],
                    item["percent"],
                    f"{res['eval_accuracy']:.4f}",
                    f"{res['eval_f1_macro']:.4f}",
                    f"{res['eval_precision_macro']:.4f}",
                    f"{res['eval_recall_macro']:.4f}"
                ])
            print(f"  -> Acc: {res['eval_accuracy']:.4f} | F1: {res['eval_f1_macro']:.4f}")

    print(f"\nEvaluation Complete. Results saved to {csv_file}")

if __name__ == "__main__":
    main()
