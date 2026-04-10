# **Methodology and Results Report Skeleton: NeFT for Multilingual Encoder Models**

## **1. Title**

**Neuron-Level Fine-Tuning for Multilingual Encoder Models: Methodology and Comparative Results across POS, NER, and Sentiment Tasks**

---

## **2. Objective**

[INSERT OBJECTIVE HERE]

**Suggested coverage:**

* Briefly state that the report documents the experimental workflow and final comparative results for encoder-based NeFT experiments.
* Mention the three task families: POS, NER, and Sentiment.
* Mention the two encoder models: mBERT and XLM-R.
* Mention the three compared methods: Baseline, NeFT, and Probeless. ([GitHub][1])

---

## **3. Methodology**

### **3.1 Project Overview**

[INSERT TEXT HERE]

**Suggested content points:**

* Introduce the project as an encoder adaptation of the NeFT workflow.
* State the overall pipeline:

  1. Baseline fine-tuning
  2. NeFT neuron selection / mask generation
  3. Probeless mask preparation
  4. Masked fine-tuning
  5. Evaluation and result aggregation
* Mention that the repo is organized around dedicated folders for baseline training, neuron discovery, probeless mask preparation, masked training, evaluation, rankings, and results. ([GitHub][1])

**Figure placeholder:**

* **Figure 1. Overall Experimental Pipeline**

  * [INSERT FIGURE HERE]

---

### **3.2 Experimental Scope**

[INSERT TEXT HERE]

**Suggested content points:**

* Tasks covered:

  * POS tagging
  * Named Entity Recognition
  * Sentiment classification
* Models covered:

  * mBERT (`bert-base-multilingual-cased`)
  * XLM-R (`xlm-roberta-base`)
* Methods compared:

  * Baseline
  * NeFT
  * Probeless
* Percent settings used for masked training:

  * 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
* State that experiments are organized through experiment keys in `config.py`. ([GitHub][1])

**Table placeholder:**

* **Table 1. Experimental Scope Summary**

  * [INSERT TABLE HERE]

---

### **3.3 Data and Task Setup**

[INSERT TEXT HERE]

**Suggested subsections:**

#### **3.3.1 POS and NER Data Setup**

[INSERT TEXT HERE]

* Describe that token-classification tasks use CoNLL-U-style train/dev/test files.
* Mention that the scripts read tokens from column 2 and labels from column 4.
* Note that the same token-classification pipeline is used for POS and NER, with task-specific datasets and experiment keys. ([GitHub][1])

#### **3.3.2 Sentiment Data Setup**

[INSERT TEXT HERE]

* Describe that sentiment experiments use CSV input with `text` and `label` columns.
* Mention the deterministic train/validation/test split.
* Note the documented 70/15/15 split through `data_utils.py`. ([GitHub][1])

#### **3.3.3 Experiment Registry and Configuration**

[INSERT TEXT HERE]

* Mention `config.py` as the central registry for dataset paths, ranking paths, mask output paths, model assignment, and experiment definitions. ([GitHub][1])

**Table placeholders:**

* **Table 2. Task-Wise Dataset and Input Format Summary**

  * [INSERT TABLE HERE]
* **Table 3. Experiment Registry Summary**

  * [INSERT TABLE HERE]

---

### **3.4 Baseline Fine-Tuning**

[INSERT TEXT HERE]

**Suggested content points:**

* Describe baseline training as the fully trainable reference model for each experiment.
* Separate the workflow for:

  * POS / NER baseline training via `baseline/train_baseline.py`
  * Sentiment baseline training via `baseline/train_baseline_sentiment.py`
* Mention that baseline checkpoints are saved under `./output/{lang}-baseline/`. ([GitHub][1])

**Table placeholder:**

* **Table 4. Baseline Fine-Tuning Scripts and Output Locations**

  * [INSERT TABLE HERE]

---

### **3.5 NeFT Neuron Selection / Mask Generation**

[INSERT TEXT HERE]

**Suggested content points:**

* Describe NeFT mask generation using `find_neuron/find_neuron.py`.
* State that the process compares pretrained encoder weights with the baseline checkpoint.
* Mention that generated outputs include neuron-scoring artifacts and percentage-specific mask files.
* Mention the documented output path:

  * `./masks/{lang}/baseline_neft/`
* Mention mask files for 5% through 50%. ([GitHub][1])

**Figure placeholder:**

* **Figure 2. NeFT Mask Generation Workflow**

  * [INSERT FIGURE HERE]

**Table placeholder:**

* **Table 5. NeFT Mask Files by Percentage Setting**

  * [INSERT TABLE HERE]

---

### **3.6 Probeless Mask Preparation**

[INSERT TEXT HERE]

**Suggested content points:**

* Describe Probeless mask creation using `prepare_probeless/prepare_masks.py`.
* State that ranking files are converted into binary masks for the predefined percentage settings.
* Mention the documented mask outputs:

  * `5probeless_mask.pt` through `50probeless_mask.pt`
* Mention the output path under `./masks/{lang}/`. ([GitHub][1])

**Figure placeholder:**

* **Figure 3. Probeless Mask Preparation Workflow**

  * [INSERT FIGURE HERE]

**Table placeholder:**

* **Table 6. Probeless Mask Files by Percentage Setting**

  * [INSERT TABLE HERE]

---

### **3.7 Masked Fine-Tuning**

[INSERT TEXT HERE]

**Suggested subsections:**

#### **3.7.1 NeFT Masked Training**

[INSERT TEXT HERE]

* Mention:

  * `train_neuron/train_neft.py` for POS / NER
  * `train_neuron/train_neft_sentiment.py` for Sentiment
* Note checkpoint pattern:

  * `./output/{lang}-neft/{p}percent/` ([GitHub][1])

#### **3.7.2 Probeless Masked Training**

[INSERT TEXT HERE]

* Mention:

  * `train_neuron/train_probeless.py` for POS / NER
  * `train_neuron/train_probeless_sentiment.py` for Sentiment
* Note checkpoint pattern:

  * `./output/{lang}-probeless/{p}percent/` ([GitHub][1])

#### **3.7.3 Percentage Sweep Design**

[INSERT TEXT HERE]

* State that all masked experiments are run for:

  * 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 percent. ([GitHub][1])

**Table placeholders:**

* **Table 7. Masked Fine-Tuning Scripts and Output Directories**

  * [INSERT TABLE HERE]
* **Table 8. Percentage Sweep Definition**

  * [INSERT TABLE HERE]

---

### **3.8 Evaluation Protocol**

[INSERT TEXT HERE]

**Suggested subsections:**

#### **3.8.1 POS / NER Evaluation**

[INSERT TEXT HERE]

* Mention evaluation through `evaluation/evaluate_model.py`
* Mention output summary path:

  * `./Results/scores/evaluation_summary_{lang}.csv`
* Mention reported metrics:

  * Accuracy
  * Macro F1
  * Macro Precision
  * Macro Recall ([GitHub][1])

#### **3.8.2 Sentiment Evaluation**

[INSERT TEXT HERE]

* Mention evaluation through `evaluation/evaluate_model_sentiment.py`
* Mention output summary path:

  * `./Results/scores/evaluation_summary_{lang}.csv`
* Mention reported metrics:

  * Accuracy
  * Macro F1
  * Weighted F1
  * Macro Precision
  * Macro Recall ([GitHub][1])

#### **3.8.3 Plotting and Visualization**

[INSERT TEXT HERE]

* Mention `evaluation/plot.py`
* Mention plot output path:

  * `./Results/metric_plots/{lang}/` ([GitHub][1])

**Table placeholder:**

* **Table 9. Evaluation Scripts, Metrics, and Output Files**

  * [INSERT TABLE HERE]

**Figure placeholder:**

* **Figure 4. Metric Plot Template**

  * [INSERT FIGURE HERE]

---

## **4. Results**

### **4.1 Per-Experiment Raw Result Tables**

[INSERT TEXT HERE]

**Suggested organization:**

* One raw table per experiment key or per language-model-task block
* Include:

  * Method
  * Percent
  * Accuracy
  * Macro F1
  * Macro Precision
  * Macro Recall
  * Weighted F1 where applicable

**Table placeholders:**

* **Table 10. Raw Results for POS Experiments**

  * [INSERT TABLE HERE]
* **Table 11. Raw Results for NER Experiments**

  * [INSERT TABLE HERE]
* **Table 12. Raw Results for Sentiment Experiments**

  * [INSERT TABLE HERE]

---

### **4.2 Cross-Method Comparison Tables**

[INSERT TEXT HERE]

**Suggested organization:**

* Compare Baseline vs NeFT vs Probeless
* Group by:

  * Task
  * Model
  * Language
* Include baseline as the 100% full fine-tuning reference and masked methods as percentage-based variants

**Table placeholders:**

* **Table 13. Cross-Method Comparison for POS**

  * [INSERT TABLE HERE]
* **Table 14. Cross-Method Comparison for NER**

  * [INSERT TABLE HERE]
* **Table 15. Cross-Method Comparison for Sentiment**

  * [INSERT TABLE HERE]

**Figure placeholders:**

* **Figure 5. Cross-Method Performance Comparison across Percentage Settings**

  * [INSERT FIGURE HERE]
* **Figure 6. Task-Wise Comparison between mBERT and XLM-R**

  * [INSERT FIGURE HERE]

---

### **4.3 Best-Performing Configuration Tables**

[INSERT TEXT HERE]

**Suggested content points:**

* Identify the best configuration for each:

  * Task
  * Model
  * Method
  * Language
* Include:

  * Best percentage
  * Best metric value
  * Corresponding checkpoint or experiment identifier

**Table placeholders:**

* **Table 16. Best-Performing Configuration by Task and Model**

  * [INSERT TABLE HERE]
* **Table 17. Best-Performing NeFT Configurations**

  * [INSERT TABLE HERE]
* **Table 18. Best-Performing Probeless Configurations**

  * [INSERT TABLE HERE]

---

### **4.4 Key Quantitative Observations**

[INSERT TEXT HERE]

**Suggested structure:**

* **Observation 1.** [INSERT OBSERVATION HERE]
* **Observation 2.** [INSERT OBSERVATION HERE]
* **Observation 3.** [INSERT OBSERVATION HERE]
* **Observation 4.** [INSERT OBSERVATION HERE]
* **Observation 5.** [INSERT OBSERVATION HERE]

**Suggested focus areas:**

* Relative performance of NeFT vs Probeless
* Gap from Baseline to masked variants
* Best-performing percent ranges
* Task sensitivity differences
* Model-wise behavior: mBERT vs XLM-R

---

## **5. Conclusion Placeholder**

[INSERT CONCLUSION HERE]

**Suggested content points:**

* Summarize the main methodological contribution
* Summarize overall comparison outcomes
* State whether masked fine-tuning remained competitive with full fine-tuning
* Mention possible future extensions

---

## **6. Appendix**

### **A. Experiment Inventory**

[INSERT TABLE HERE]

**Suggested columns:**

* Experiment Key
* Task
* Model
* Language
* Method
* Dataset Path
* Ranking File
* Mask Path
* Output Path
* Evaluation Summary File

**Suggested table number:**

* **Table A1. Complete Experiment Inventory**

---

### **B. File/Path Mapping**

[INSERT TABLE HERE]

**Suggested content points:**

* `baseline/train_baseline.py`
* `baseline/train_baseline_sentiment.py`
* `find_neuron/find_neuron.py`
* `prepare_probeless/prepare_masks.py`
* `train_neuron/train_neft.py`
* `train_neuron/train_neft_sentiment.py`
* `train_neuron/train_probeless.py`
* `train_neuron/train_probeless_sentiment.py`
* `evaluation/evaluate_model.py`
* `evaluation/evaluate_model_sentiment.py`
* `evaluation/plot.py`
* `config.py`
* `data_utils.py` ([GitHub][1])

**Suggested columns:**

* File / Script
* Role in Pipeline
* Input
* Output
* Relevant Section in Report

**Suggested table number:**

* **Table B1. Repository File and Path Mapping**

---

### **C. Metric Definitions**

[INSERT TABLE HERE]

**Suggested rows:**

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1
* Weighted F1

**Suggested columns:**

* Metric
* Definition
* Applicable Tasks
* Interpretation in This Report

**Suggested table number:**

* **Table C1. Metric Definitions**

---

## **List of Tables Placeholder**

[INSERT AUTO-GENERATED OR MANUAL TABLE LIST HERE]

---

## **List of Figures Placeholder**

[INSERT AUTO-GENERATED OR MANUAL FIGURE LIST HERE]

[1]: https://github.com/PharasiG/NeFT "GitHub - PharasiG/NeFT: [COLING 2025] Let's Focus on Neuron: Neuron-Level Supervised Fine-tuning for Large Language Model · GitHub"
