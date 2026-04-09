import os
import sys
import argparse
import datasets
import transformers
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
    EarlyStoppingCallback
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

# =========================
# CoNLL-U Parser
# =========================
def read_conllu(file_path: str):
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

            parts = line.split("\t")
            if "-" in parts[0] or "." in parts[0]:
                continue

            tokens.append(parts[1])      # FORM
            tags.append(parts[3])        # Label column

    return sentences, labels


# =========================
# Main
# =========================
def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py")
    args = parser.parse_args()

    # 2. Load Config & Validate
    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]
    data_dir = os.path.join(config.DATASET_ROOT, cfg["dataset_folder"])
    model_name = cfg["model_name"]

    print(f"Start Baseline Training for: {args.lang}")
    print(f"Using model: {model_name}")
    transformers.set_seed(42)

    # 3. Load Data (Dynamic Paths)
    train_file = os.path.join(data_dir, f"{cfg['conllu_prefix']}-ud-train.conllu")
    dev_file = os.path.join(data_dir, f"{cfg['conllu_prefix']}-ud-dev.conllu")

    train_tokens, train_tags = read_conllu(train_file)
    dev_tokens, dev_tags = read_conllu(dev_file)

    # Label mapping
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

    # HF Dataset
    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"tokens": train_tokens, "labels": train_tags}),
        "validation": datasets.Dataset.from_dict({"tokens": dev_tokens, "labels": dev_tags}),
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=256,
        )

        aligned_labels = []
        for i in range(len(tokenized["input_ids"])):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[examples["labels"][i][word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # 4. Training Arguments
    output_dir = f"./output/{args.lang}-baseline"

    training_args = TrainingArguments(
        output_dir=output_dir,
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
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()
    trainer.save_model()
    print(f"Baseline model saved to {output_dir}")


if __name__ == "__main__":
    main()