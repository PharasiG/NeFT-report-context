# NeFT for Multilingual POS, NER, and Sentiment on Encoder Models

This repository adapts **NeFT (Neuron-Level Fine-Tuning)** to **encoder-based multilingual models** for three task families:

- **POS tagging**
- **Named Entity Recognition (NER)**
- **Sentiment classification**

The codebase supports both **mBERT** (`bert-base-multilingual-cased`) and **XLM-R** (`xlm-roberta-base`) through experiment keys defined in `config.py`.

---

## What this repo does

At a high level, the project follows this workflow:

1. **Baseline fine-tuning**
   - Train a full baseline model for the selected task and language.
2. **Neuron selection / mask generation**
   - **NeFT**: derive masks by comparing pretrained and fine-tuned encoder weights.
   - **Probeless**: convert a precomputed ranking file into binary masks.
3. **Masked fine-tuning**
   - Retrain while updating only selected FFN neurons.
4. **Evaluation and plotting**
   - Evaluate baseline, NeFT, and Probeless checkpoints.
   - Plot metric curves across neuron percentages.

---

## Repository structure

```text
NeFT/
├── baseline/
│   ├── train_baseline.py              # baseline training for POS / NER
│   └── train_baseline_sentiment.py    # baseline training for sentiment
├── train_neuron/
│   ├── train_neft.py                  # NeFT masked training for POS / NER
│   ├── train_neft_sentiment.py        # NeFT masked training for sentiment
│   ├── train_probeless.py             # Probeless masked training for POS / NER
│   └── train_probeless_sentiment.py   # Probeless masked training for sentiment
├── find_neuron/
│   └── find_neuron.py                 # derive NeFT masks from pretrained vs baseline weights
├── prepare_probeless/
│   └── prepare_masks.py               # convert ranking pickle into probeless masks
├── evaluation/
│   ├── evaluate_model.py              # evaluate POS / NER checkpoints
│   ├── evaluate_model_sentiment.py    # evaluate sentiment checkpoints
│   └── plot.py                        # plot baseline vs NeFT vs Probeless metrics
├── cl_head_base/
│   └── baseline.py                    # older linear-probe POS baseline script
├── ranking/                           # stored ranking files (.pkl)
├── Results/                           # scores, plots, and experiment outputs
├── config.py                          # central experiment registry and paths
├── data_utils.py                      # deterministic sentiment train/val/test split helper
└── README.md
```

---

## Core scripts

### `config.py`
This is the control center of the repo.

It defines:

- `DATASET_ROOT`
- `RANKING_ROOT`
- `OUTPUT_MASK_DIR`
- model constants for **mBERT** and **XLM-R**
- `EXPERIMENTS`, which map each experiment key to its dataset, ranking file, and model

Current experiment groups include:

- **mBERT POS**: `mBERT_hindi_pos`, `mBERT_arabic_pos`, `mBERT_Chinese_pos`, `mBERT_french_pos`
- **mBERT NER**: `mBERT_hindi_ner`, `mBERT_arabic_ner`, `mBERT_chinese_ner`, `mBERT_french_ner`
- **XLM-R POS**: `XLM_hindi_pos`, `XLM_arabic_pos`, `XLM_chinese_pos`, `XLM_french_pos`
- **XLM-R NER**: `XLM_hindi_ner`, `XLM_arabic_ner`, `XLM_chinese_ner`, `XLM_french_ner`
- **mBERT Sentiment**: `mBERT_hindi_sentiment`, `mBERT_arabic_sentiment`, `mBERT_chinese_sentiment`, `mBERT_french_sentiment`
- **XLM-R Sentiment**: `XLM_hindi_sentiment`, `XLM_arabic_sentiment`, `XLM_chinese_sentiment`, `XLM_french_sentiment`

### `baseline/train_baseline.py`
Baseline trainer for **POS / NER**.

- reads CoNLL-U-style train/dev files
- uses the model defined by the chosen experiment key in `config.py`
- saves checkpoints to:

```text
./output/{lang}-baseline/
```

### `baseline/train_baseline_sentiment.py`
Baseline trainer for **sentiment classification**.

- reads a CSV dataset through `data_utils.py`
- uses a deterministic **70 / 15 / 15** split
- trains `AutoModelForSequenceClassification`
- saves checkpoints to:

```text
./output/{lang}-baseline/
```

### `find_neuron/find_neuron.py`
Generates **NeFT masks** by comparing pretrained encoder weights and the baseline checkpoint.

Outputs are saved under:

```text
./masks/{lang}/baseline_neft/
```

Generated files include:

- `cos_out.csv`
- `5neft_mask.pt`
- `10neft_mask.pt`
- ...
- `50neft_mask.pt`

### `prepare_probeless/prepare_masks.py`
Converts a ranking pickle into **Probeless masks**.

Outputs are saved under:

```text
./masks/{lang}/
```

Generated files include:

- `5probeless_mask.pt`
- `10probeless_mask.pt`
- ...
- `50probeless_mask.pt`

### `train_neuron/train_neft.py`
Runs **NeFT masked training** for **POS / NER**.

Saves checkpoints to:

```text
./output/{lang}-neft/{p}percent/
```

### `train_neuron/train_probeless.py`
Runs **Probeless masked training** for **POS / NER**.

Saves checkpoints to:

```text
./output/{lang}-probeless/{p}percent/
```

### `train_neuron/train_neft_sentiment.py`
Runs **NeFT masked training** for **sentiment**.

Saves checkpoints to:

```text
./output/{lang}-neft/{p}percent/
```

### `train_neuron/train_probeless_sentiment.py`
Runs **Probeless masked training** for **sentiment**.

Saves checkpoints to:

```text
./output/{lang}-probeless/{p}percent/
```

### `evaluation/evaluate_model.py`
Evaluates **POS / NER** checkpoints on the test split.

Writes:

```text
./Results/scores/evaluation_summary_{lang}.csv
```

Metrics:

- Accuracy
- Macro F1
- Macro Precision
- Macro Recall

### `evaluation/evaluate_model_sentiment.py`
Evaluates **sentiment** checkpoints on the held-out test split.

Writes:

```text
./Results/scores/evaluation_summary_{lang}.csv
```

Metrics:

- Accuracy
- Macro F1
- Weighted F1
- Macro Precision
- Macro Recall

### `evaluation/plot.py`
Plots comparison curves from an evaluation CSV.

Saves plots to:

```text
./Results/metric_plots/{lang}/
```

This script is now **CLI-driven**:

```bash
python evaluation/plot.py --lang XLM_hindi_pos
```

---

## Environment setup

Use **Python 3.10 or 3.11**.

### Clone and create environment

```bash
git clone -b LLR https://github.com/PharasiG/NeFT.git
cd NeFT

python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install torch transformers datasets accelerate evaluate seqeval scikit-learn pandas matplotlib
```

If you are using GPU, install the correct CUDA build of PyTorch for your system.

---

## Data layout expected by the code

### 1. POS / NER datasets

The token-classification scripts expect **CoNLL-U-style** files named like:

```text
{prefix}-ud-train.conllu
{prefix}-ud-dev.conllu
{prefix}-ud-test.conllu
```

Example:

```text
data/
└── UD_Hindi-HDTB/
    ├── hi_hdtb-ud-train.conllu
    ├── hi_hdtb-ud-dev.conllu
    └── hi_hdtb-ud-test.conllu
```

The scripts read:

- token from **column 2**
- label/tag from **column 4**

They also skip:

- multiword token rows such as `1-2`
- empty node rows such as `3.1`

So NER datasets should also follow the same CoNLL-U-like structure used by the code.

### 2. Sentiment datasets

The sentiment pipeline expects a CSV file with at least these columns:

```text
text,label
```

The dataset path is defined in `config.py` through:

- `dataset_folder`
- `data_file`

Example config entry:

```python
"XLM_hindi_sentiment": {
    "model_name": XLMR_MODEL_NAME,
    "ranking_rel_path": "XLM_hindi_sentiment_ranking",
    "dataset_folder": "sentiment",
    "data_file": "cleaned_hindi_data.csv"
}
```

`data_utils.py` performs a deterministic stratified split:

- **70% train**
- **15% validation**
- **15% test**

---

## Configure paths before running

Open `config.py` and verify:

```python
DATASET_ROOT = "./data"
RANKING_ROOT = "./ranking"
OUTPUT_MASK_DIR = "./masks"
```

Then make sure the experiment key you want to run exists in `EXPERIMENTS`.

---

## How to run the code

## A. POS / NER pipeline

### Step 1: Train the baseline

Example: XLM-R POS

```bash
python baseline/train_baseline.py --lang XLM_hindi_pos
```

Example: mBERT NER

```bash
python baseline/train_baseline.py --lang mBERT_hindi_ner
```

This creates:

```text
./output/{lang}-baseline/
```

### Step 2: Generate NeFT masks from the baseline

```bash
python find_neuron/find_neuron.py --lang XLM_hindi_pos
```

This creates:

```text
./masks/XLM_hindi_pos/baseline_neft/
├── cos_out.csv
├── 5neft_mask.pt
├── 10neft_mask.pt
├── ...
└── 50neft_mask.pt
```

### Step 3: Generate Probeless masks

```bash
python prepare_probeless/prepare_masks.py --lang XLM_hindi_pos
```

This creates:

```text
./masks/XLM_hindi_pos/
├── 5probeless_mask.pt
├── 10probeless_mask.pt
├── ...
└── 50probeless_mask.pt
```

### Step 4: Train NeFT models

```bash
python train_neuron/train_neft.py --lang XLM_hindi_pos
```

This trains percentages:

```text
[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
```

Outputs:

```text
./output/XLM_hindi_pos-neft/5percent/
...
./output/XLM_hindi_pos-neft/50percent/
```

### Step 5: Train Probeless models

```bash
python train_neuron/train_probeless.py --lang XLM_hindi_pos
```

Outputs:

```text
./output/XLM_hindi_pos-probeless/5percent/
...
./output/XLM_hindi_pos-probeless/50percent/
```

### Step 6: Evaluate

```bash
python evaluation/evaluate_model.py --lang XLM_hindi_pos
```

This writes:

```text
./Results/scores/evaluation_summary_XLM_hindi_pos.csv
```

### Step 7: Plot metrics

```bash
python evaluation/plot.py --lang XLM_hindi_pos
```

Plots are saved to:

```text
./Results/metric_plots/XLM_hindi_pos/
```

---

## B. Sentiment pipeline

### Step 1: Train the baseline

Example: XLM-R Hindi sentiment

```bash
python baseline/train_baseline_sentiment.py --lang XLM_hindi_sentiment
```

Example: mBERT Arabic sentiment

```bash
python baseline/train_baseline_sentiment.py --lang mBERT_arabic_sentiment
```

This creates:

```text
./output/{lang}-baseline/
```

### Step 2: Generate NeFT masks from the baseline

```bash
python find_neuron/find_neuron.py --lang XLM_hindi_sentiment
```

### Step 3: Generate Probeless masks

```bash
python prepare_probeless/prepare_masks.py --lang XLM_hindi_sentiment
```

### Step 4: Train NeFT sentiment models

```bash
python train_neuron/train_neft_sentiment.py --lang XLM_hindi_sentiment
```

### Step 5: Train Probeless sentiment models

```bash
python train_neuron/train_probeless_sentiment.py --lang XLM_hindi_sentiment
```

### Step 6: Evaluate

```bash
python evaluation/evaluate_model_sentiment.py --lang XLM_hindi_sentiment
```

This writes:

```text
./Results/scores/evaluation_summary_XLM_hindi_sentiment.csv
```

### Step 7: Plot metrics

```bash
python evaluation/plot.py --lang XLM_hindi_sentiment
```

---

## End-to-end examples

### XLM-R POS

```bash
python baseline/train_baseline.py --lang XLM_hindi_pos
python find_neuron/find_neuron.py --lang XLM_hindi_pos
python prepare_probeless/prepare_masks.py --lang XLM_hindi_pos
python train_neuron/train_neft.py --lang XLM_hindi_pos
python train_neuron/train_probeless.py --lang XLM_hindi_pos
python evaluation/evaluate_model.py --lang XLM_hindi_pos
python evaluation/plot.py --lang XLM_hindi_pos
```

### XLM-R sentiment

```bash
python baseline/train_baseline_sentiment.py --lang XLM_hindi_sentiment
python find_neuron/find_neuron.py --lang XLM_hindi_sentiment
python prepare_probeless/prepare_masks.py --lang XLM_hindi_sentiment
python train_neuron/train_neft_sentiment.py --lang XLM_hindi_sentiment
python train_neuron/train_probeless_sentiment.py --lang XLM_hindi_sentiment
python evaluation/evaluate_model_sentiment.py --lang XLM_hindi_sentiment
python evaluation/plot.py --lang XLM_hindi_sentiment
```

---

## Important assumptions and caveats

1. **Mask code assumes encoder FFN geometry of 12 layers, 768 hidden size, and 3072 intermediate size.**
   This matches base mBERT and base XLM-R, but it is not model-agnostic.

2. **`find_neuron.py` currently derives masks only from `output.dense.weight`.**
   It does not create masks for other parameter groups.

3. **`prepare_masks.py` assumes ranking values map to flattened neuron ids.**
   It also removes layer 0 and shifts layer indices by `-1`, so ranking files must match that convention.

4. **Evaluation scripts skip checkpoints that cannot be loaded.**
   This is useful when only some percentages have been trained.

5. **Plotting expects the CSV format generated by the repo's evaluation scripts.**

---

## Legacy script

### `cl_head_base/baseline.py`
This is an older **linear-probe POS baseline** script that freezes the encoder and trains only the classification head. It is separate from the main POS / NER / sentiment workflow documented above.

---

## Citation

If you use the original NeFT idea, please cite the paper:

```bibtex
@inproceedings{xu-etal-2025-lets,
  title = "Let{'}s Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model",
  author = "Xu, Haoyun and Zhan, Runzhe and Ma, Yingpeng and Wong, Derek F. and Chao, Lidia S.",
  booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
  year = "2025",
  pages = "9393--9406"
}
```

---

## Quick command summary

```bash
# POS / NER baseline
python baseline/train_baseline.py --lang XLM_hindi_pos

# sentiment baseline
python baseline/train_baseline_sentiment.py --lang XLM_hindi_sentiment

# NeFT masks
python find_neuron/find_neuron.py --lang XLM_hindi_pos

# Probeless masks
python prepare_probeless/prepare_masks.py --lang XLM_hindi_pos

# NeFT training
python train_neuron/train_neft.py --lang XLM_hindi_pos

# Probeless training
python train_neuron/train_probeless.py --lang XLM_hindi_pos

# sentiment NeFT training
python train_neuron/train_neft_sentiment.py --lang XLM_hindi_sentiment

# sentiment Probeless training
python train_neuron/train_probeless_sentiment.py --lang XLM_hindi_sentiment

# evaluation
python evaluation/evaluate_model.py --lang XLM_hindi_pos
python evaluation/evaluate_model_sentiment.py --lang XLM_hindi_sentiment

# plotting
python evaluation/plot.py --lang XLM_hindi_pos
```
