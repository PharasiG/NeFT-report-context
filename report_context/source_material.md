### `config.py`

* **1. Purpose**

  * Central experiment registry for the repo: defines root directories (`./data`, `./ranking`, `./masks`), model constants for mBERT and XLM-R, and the `EXPERIMENTS` dictionary that maps each experiment key to its model, dataset location, and ranking file. ([GitHub][1])
* **2. Inputs**

  * No CLI input; downstream scripts read `DATASET_ROOT`, `RANKING_ROOT`, `OUTPUT_MASK_DIR`, and per-experiment fields such as `model_name`, `dataset_folder`, `conllu_prefix`, and `data_file`. ([GitHub][1])
* **3. Processing logic**

  * Stores task-specific experiment metadata for POS/NER in CoNLL-U-style folders and Sentiment in the `sentiment` folder with CSV filenames; also ties each experiment key to a specific probeless ranking file path relative to `./ranking`. ([GitHub][1])
* **4. Outputs**

  * Exposes configuration values only; it does not itself create masks, checkpoints, CSV summaries, or plots. ([GitHub][1])
* **5. Important hyperparameters or fixed settings**

  * Fixed model constants are `bert-base-multilingual-cased` and `xlm-roberta-base`; root paths are hardcoded as relative paths unless manually changed. ([GitHub][1])
* **6. What should be mentioned in a formal report**

  * Report the exact experiment registry role: each run is identified by a config key, and that key determines the model backbone, dataset source, and ranking source used by the rest of the pipeline. ([GitHub][1])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim that `config.py` centrally defines output checkpoint paths, evaluation CSV paths, or the percentage sweep; those are constructed or hardcoded in other scripts, not in this file. ([GitHub][1])

### `baseline/train_baseline.py`

* **1. Purpose**

  * Trains the token-classification baseline model for POS/NER-style experiments. ([GitHub][2])
* **2. Inputs**

  * Takes `--lang`, validates it against `config.EXPERIMENTS`, then reads `train` and `dev` files from `./data/{dataset_folder}/{conllu_prefix}-ud-{split}.conllu`. ([GitHub][2])
* **3. Processing logic**

  * Reads CoNLL-U-style data by taking token text from column 2 (`parts[1]`) and labels from column 4 (`parts[3]`), skips comment lines and multiword/decimal IDs, builds label mappings from training labels, tokenizes with `is_split_into_words=True`, assigns labels only to the first subtoken, and uses `seqeval` for validation metrics. ([GitHub][2])
* **4. Outputs**

  * Saves the trained baseline checkpoint to `./output/{lang}-baseline`. ([GitHub][2])
* **5. Important hyperparameters or fixed settings**

  * Seed `42`; `max_length=256`; `num_train_epochs=20`; train batch size `8`; eval batch size `16`; learning rate `2e-5`; eval/save every epoch; `load_best_model_at_end=True`; best-model metric `f1`; `logging_steps=500`; early stopping patience `5`; `save_total_limit=2`; dataset tokenization uses `num_proc=4`. ([GitHub][2])
* **6. What should be mentioned in a formal report**

  * State clearly that this stage is full fine-tuning on train/dev data only for token classification, with subtoken label alignment and validation tracked using precision, recall, F1, and accuracy from `seqeval`. ([GitHub][2])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim test-set evaluation in this script, CRF decoding, architecture changes, or any label source other than column 4 of the input files. ([GitHub][2])

### `baseline/train_baseline_sentiment.py`

* **1. Purpose**

  * Trains the baseline sequence-classification model for sentiment experiments. ([GitHub][3])
* **2. Inputs**

  * Takes `--lang`, loads experiment metadata from `config.py`, and calls `load_sentiment_splits(cfg, config.DATASET_ROOT)` to obtain train/validation/test partitions from the configured sentiment CSV. ([GitHub][3])
* **3. Processing logic**

  * Uses a deterministic stratified split utility with seed `42`, overall ratios `70/15/15`, drops rows with missing `text` or `label`, casts labels to `int`, builds label maps from the training split, tokenizes raw text with `max_length=256`, and evaluates validation checkpoints using accuracy and weighted F1. ([GitHub][3])
* **4. Outputs**

  * Saves the trained baseline checkpoint to `./output/{lang}-baseline`. ([GitHub][3])
* **5. Important hyperparameters or fixed settings**

  * Seed `42`; `max_length=256`; `num_train_epochs=20`; train batch size `16`; eval batch size `32`; learning rate `2e-5`; eval/save every epoch; `load_best_model_at_end=True`; best-model metric `f1`; `logging_steps=200`; early stopping patience `5`; `save_total_limit=2`. ([GitHub][3])
* **6. What should be mentioned in a formal report**

  * Mention that sentiment data is split once, deterministically, into train/validation/test, and that only train/validation are used in this stage while the test split remains held out. ([GitHub][3])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim macro F1 reporting during training, test-set-based model selection, or any custom classifier head beyond `AutoModelForSequenceClassification`. ([GitHub][3])

### `find_neuron/find_neuron.py`

* **1. Purpose**

  * Generates NeFT masks by comparing the pretrained encoder to the baseline-fine-tuned checkpoint for a given experiment key. ([GitHub][4])
* **2. Inputs**

  * Takes `--lang`, reads `model_name` from config as the reference pretrained model, expects the baseline checkpoint at `./output/{lang}-baseline`, and writes artifacts under `./masks/{lang}/baseline_neft/`. ([GitHub][4])
* **3. Processing logic**

  * Loads both models with `AutoModel`, iterates over `model.encoder.layer`, extracts only `output.dense.weight`, computes per-row cosine similarity between pretrained and baseline weights, saves `cos_out.csv`, converts the requested percentage into `int(12*768*p/100)` neurons, finds a threshold from the sorted cosine scores, selects entries with `score < threshold`, and writes binary mask tensors keyed as `{layer_idx}_out`. ([GitHub][4])
* **4. Outputs**

  * Produces `cos_out.csv` plus one mask file per percentage: `5neft_mask.pt` through `50neft_mask.pt` in `./masks/{lang}/baseline_neft/`. ([GitHub][4])
* **5. Important hyperparameters or fixed settings**

  * Fixed assumptions: `12` layers, `768` hidden size, `3072` intermediate size, percent list `[5,10,15,20,25,30,35,40,45,50]`; if `cos_out.csv` already exists, cosine computation is skipped. ([GitHub][4])
* **6. What should be mentioned in a formal report**

  * Say that NeFT mask generation in this repo is based on cosine-difference ranking between pretrained and baseline encoder FFN output-projection weights, and that the masks are binary selectors over flattened `output.dense.weight` gradient positions. ([GitHub][4])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim that this script analyzes activations, gradients, attention weights, classifier heads, or `intermediate.dense` / W_in weights; it only compares encoder `output.dense.weight`. Do not claim exact top-k selection under ties, because selection is based on `score < threshold`, not guaranteed exact equality to the requested count. ([GitHub][4])

### `prepare_probeless/prepare_masks.py`

* **1. Purpose**

  * Converts stored probeless neuron rankings into binary mask files for the masked-training stage. ([GitHub][5])
* **2. Inputs**

  * Takes `--lang`, resolves the ranking pickle from `./ranking/{ranking_rel_path}`, and uses the configured output directory `./masks/{lang}/`. ([GitHub][5])
* **3. Processing logic**

  * Loads the pickle, wraps it into a one-column DataFrame, derives `layer_idx = value // 768` and `neuron_idx = value % 768`, removes entries assigned to layer `0`, shifts remaining layer indices down by `1`, takes the first `y = int(9216 * p/100)` rows for each percentage, groups neuron IDs by layer, and creates flattened binary masks keyed as `{layer}_out` with length `768*3072`. ([GitHub][5])
* **4. Outputs**

  * Saves one file per percentage, `5probeless_mask.pt` through `50probeless_mask.pt`, under `./masks/{lang}/`. ([GitHub][5])
* **5. Important hyperparameters or fixed settings**

  * Fixed constants are `NEURONS_PER_LAYER=768`, `TOTAL_NEURONS=9216`, `intermediate_size=3072`, `hidden_size=768`, and percentages `[5,10,15,20,25,30,35,40,45,50]`. ([GitHub][5])
* **6. What should be mentioned in a formal report**

  * Mention that the probeless stage converts precomputed global neuron rankings into binary output-projection masks and that this script assumes the ranking file is already ordered, because it uses `head(y)` directly. ([GitHub][5])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim that this script recomputes ranking scores, sorts the ranking internally, preserves original layer numbering, or builds masks for `intermediate.dense` / W_in weights. ([GitHub][5])

### `train_neuron/train_neft.py`

* **1. Purpose**

  * Runs NeFT-style masked fine-tuning for POS/NER token-classification experiments across all percentage settings. ([GitHub][6])
* **2. Inputs**

  * Takes `--lang`, reads train/dev CoNLL-U files from config, loads masks from `./masks/{lang}/baseline_neft/{p}neft_mask.pt`, and uses the backbone in `cfg["model_name"]`. ([GitHub][6])
* **3. Processing logic**

  * Parses tokens from field 2 and labels from field 4, tokenizes with `is_split_into_words=True` and `max_length=256`, aligns labels to first subtokens, then for each percent reinitializes `AutoModelForTokenClassification` from the pretrained backbone and applies a callback that multiplies gradients of `layer.{i}.output.dense.weight` by `mask.view(768,3072)` at `on_pre_optimizer_step`. ([GitHub][6])
* **4. Outputs**

  * Writes one checkpoint directory per percentage under `./output/{lang}-neft/{p}percent`. ([GitHub][6])
* **5. Important hyperparameters or fixed settings**

  * Percent list `[5,10,15,20,25,30,35,40,45,50]`; `num_train_epochs=20`; train batch size `8`; eval batch size `16`; learning rate `2e-5`; eval/save every epoch; best-model metric `f1`; `logging_steps=500`; early stopping patience `5`; `save_total_limit=2`; gradient masking is hardcoded for `12` layers with mask reshape `(768,3072)`. ([GitHub][6])
* **6. What should be mentioned in a formal report**

  * Report that masked training is run separately for each percentage, that NeFT masks come from the baseline-comparison stage, and that the implementation masks gradients only for encoder FFN `output.dense.weight` parameters. ([GitHub][6])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim that masked training resumes from the baseline checkpoint, freezes all non-selected parameters, or masks attention / `intermediate.dense` weights. The script starts each run from the pretrained backbone and only zeroes selected gradient entries in `output.dense.weight`; other parameters remain trainable. ([GitHub][6])

### `train_neuron/train_probeless.py`

* **1. Purpose**

  * Runs probeless-mask-based fine-tuning for POS/NER token-classification experiments across all percentage settings. ([GitHub][7])
* **2. Inputs**

  * Takes `--lang`, reads train/dev CoNLL-U files from config, loads masks from `./masks/{lang}/{p}probeless_mask.pt`, and uses the configured backbone model. ([GitHub][7])
* **3. Processing logic**

  * Uses the same token parser, tokenization, label alignment, and `seqeval` validation setup as `train_neft.py`, but swaps in probeless mask files; the callback still masks gradients only on `layer.{i}.output.dense.weight` before each optimizer step. ([GitHub][7])
* **4. Outputs**

  * Saves one checkpoint directory per percentage under `./output/{lang}-probeless/{p}percent`. ([GitHub][7])
* **5. Important hyperparameters or fixed settings**

  * Same core settings as token NeFT training, except `save_total_limit=3` here instead of `2`; percentages remain `[5,10,15,20,25,30,35,40,45,50]`. ([GitHub][7])
* **6. What should be mentioned in a formal report**

  * Mention that the probeless branch reuses the same masked-fine-tuning mechanism as NeFT in this repo; the difference is only the source of the masks, not a different training engine. ([GitHub][7])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim that probeless training uses a different callback, different masked parameter family, or the baseline checkpoint as initialization; it starts from the pretrained backbone and masks only `output.dense.weight` gradients. ([GitHub][7])

### `train_neuron/train_neft_sentiment.py`

* **1. Purpose**

  * Runs NeFT-style masked fine-tuning for sentiment classification across the percentage sweep. ([GitHub][8])
* **2. Inputs**

  * Takes `--lang`, loads train/validation splits through `load_sentiment_splits`, uses masks from `./masks/{lang}/baseline_neft/{p}neft_mask.pt`, and initializes the sequence-classification model from `cfg["model_name"]`. ([GitHub][8])
* **3. Processing logic**

  * Builds label maps from training labels, tokenizes text with `max_length=256`, computes validation accuracy and weighted F1, and for each percentage reinitializes `AutoModelForSequenceClassification` and applies the same `output.dense.weight` gradient-masking callback used in token tasks. ([GitHub][8])
* **4. Outputs**

  * Saves per-percentage checkpoints to `./output/{lang}-neft/{p}percent`. ([GitHub][8])
* **5. Important hyperparameters or fixed settings**

  * Deterministic split seed `42` comes from the shared split utility; training uses `20` epochs, batch sizes `16/32`, learning rate `2e-5`, epoch-level eval/save, best-model metric `f1`, `logging_steps=200`, patience `5`, and `save_total_limit=2`. ([GitHub][8])
* **6. What should be mentioned in a formal report**

  * State that sentiment masked fine-tuning uses the same held-out split protocol as the baseline and that each percent run is an independent masked fine-tuning run starting from the pretrained backbone. ([GitHub][8])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim macro F1 optimization during training, baseline-checkpoint initialization, or full-parameter freezing; the script optimizes accuracy/weighted-F1 on validation and only masks `output.dense.weight` gradients. ([GitHub][8])

### `train_neuron/train_probeless_sentiment.py`

* **1. Purpose**

  * Runs probeless-mask-based fine-tuning for sentiment classification across the percentage sweep. ([GitHub][9])
* **2. Inputs**

  * Takes `--lang`, loads train/validation sentiment splits from the shared utility, and reads masks from `./masks/{lang}/{p}probeless_mask.pt`. ([GitHub][9])
* **3. Processing logic**

  * Uses the same data split, tokenization, label mapping, metric computation, and gradient-masking callback structure as `train_neft_sentiment.py`, but swaps in the probeless mask files. ([GitHub][9])
* **4. Outputs**

  * Saves per-percentage checkpoints to `./output/{lang}-probeless/{p}percent`. ([GitHub][9])
* **5. Important hyperparameters or fixed settings**

  * Uses `20` epochs, train/eval batch sizes `16/32`, learning rate `2e-5`, epoch-level eval/save, best-model metric `f1`, `logging_steps=200`, patience `5`, and `save_total_limit=3`; percentages are `[5,10,15,20,25,30,35,40,45,50]`. ([GitHub][9])
* **6. What should be mentioned in a formal report**

  * Mention that the probeless sentiment branch differs from the NeFT branch only in mask provenance; the training implementation itself is otherwise the same masked fine-tuning pattern. ([GitHub][9])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim a different model architecture, a different masked parameter family, or baseline-checkpoint warm start; this script also starts from the pretrained backbone and only masks `output.dense.weight` gradients. ([GitHub][9])

### `evaluation/evaluate_model.py`

* **1. Purpose**

  * Evaluates token-classification checkpoints and writes a per-experiment summary CSV for baseline, NeFT, and Probeless runs. ([GitHub][10])
* **2. Inputs**

  * Takes `--lang`, reads `./data/{dataset_folder}/{conllu_prefix}-ud-test.conllu`, and checks checkpoint paths `./output/{lang}-baseline`, `./output/{lang}-neft/{p}percent`, and `./output/{lang}-probeless/{p}percent`. ([GitHub][10])
* **3. Processing logic**

  * Parses the test file by keeping only 10-column token rows, skipping comments and multiword/decimal IDs, reading tokens from field 2 and labels from field 4, tokenizing with `max_length=256`, aligning first-subtoken labels, mapping unknown test tags with `label2id.get(tag, -100)`, and computing token-level accuracy, macro precision, macro recall, and macro F1 using `sklearn` over non-ignored positions. ([GitHub][10])
* **4. Outputs**

  * Creates `./Results/scores/evaluation_summary_{lang}.csv` with columns `Model_Type, Percent, Accuracy, F1_Macro, Precision, Recall`. ([GitHub][10])
* **5. Important hyperparameters or fixed settings**

  * Fixed `MAX_LENGTH=256`; percentage list `[5,10,15,20,25,30,35,40,45,50]`; baseline is recorded with percent `0`; models that cannot be loaded are skipped rather than crashing the whole sweep. ([GitHub][10])
* **6. What should be mentioned in a formal report**

  * Mention that final token-task result tables are generated from held-out test-set evaluations and stored in a standardized CSV format covering baseline and all masked percentages. ([GitHub][10])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim `seqeval`-based final evaluation, weighted F1 output, or an actual 100% masked evaluation sweep; despite comments saying “5% - 100%,” the implemented list stops at `50`. Also do not claim the defined `BATCH_SIZE=16` is used, because it is not passed into `TrainingArguments` in this script. ([GitHub][10])

### `evaluation/evaluate_model_sentiment.py`

* **1. Purpose**

  * Evaluates sentiment checkpoints and writes a per-experiment summary CSV for baseline, NeFT, and Probeless runs. ([GitHub][11])
* **2. Inputs**

  * Takes `--lang`, reloads the held-out test split via `load_sentiment_splits`, tokenizes it once using the configured backbone tokenizer, and checks baseline/NeFT/Probeless checkpoint paths under `./output/`. ([GitHub][11])
* **3. Processing logic**

  * Loads `AutoModelForSequenceClassification` from each checkpoint, evaluates with `Trainer`, and reports accuracy, macro precision, macro recall, macro F1, and weighted F1 computed from `sklearn` metrics; the script writes rows only for checkpoints that load successfully. ([GitHub][11])
* **4. Outputs**

  * Creates `./Results/scores/evaluation_summary_{lang}.csv` with columns `Model_Type, Percent, Accuracy, F1_Macro, F1_Weighted, Precision, Recall`. ([GitHub][11])
* **5. Important hyperparameters or fixed settings**

  * Fixed `BATCH_SIZE=32`; `MAX_LENGTH=256`; percentage list `[5,10,15,20,25,30,35,40,45,50]`; baseline is recorded with percent `0`. ([GitHub][11])
* **6. What should be mentioned in a formal report**

  * Mention that sentiment result tables are generated from the held-out test partition only, not from training or validation data, and that both macro and weighted F1 are included in the summary CSV. ([GitHub][11])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim calibration metrics, ROC-AUC, confusion matrices, or any test-time ensembling; this evaluator reports only accuracy, macro precision/recall/F1, and weighted F1 from single checkpoints. ([GitHub][11])

### `evaluation/plot.py`

* **1. Purpose**

  * Generates metric-comparison plots from the evaluation summary CSVs. ([GitHub][12])
* **2. Inputs**

  * Takes `--lang`, reads `./Results/scores/evaluation_summary_{lang}.csv`, and writes plot images to `./Results/metric_plots/{lang}`. ([GitHub][12])
* **3. Processing logic**

  * Normalizes model names to `baseline`, `neft`, and `probeless`, splits the CSV by model type, plots Probeless and NeFT as percent-wise curves, and draws the baseline as a horizontal reference line; titles include the language string by default. ([GitHub][12])
* **4. Outputs**

  * Saves one PNG per enabled metric using the filename pattern `{lang}_{metric}_comparison.png`. ([GitHub][12])
* **5. Important hyperparameters or fixed settings**

  * Default toggles enable `Accuracy`, `Precision`, `Recall`, and `F1_Macro`; `F1_Weighted` is off by default; `SHOW_PLOTS=False`, `SAVE_PLOTS=True`, `FIG_SIZE=(10, 6)`, and saved images use `dpi=300`. ([GitHub][12])
* **6. What should be mentioned in a formal report**

  * Mention that plots are post-processing artifacts generated from the summary CSVs and that the baseline is visualized as a constant reference while masked methods are shown across percentages. ([GitHub][12])
* **7. What should NOT be claimed because the code does not show it**

  * Do not claim statistical significance testing, confidence intervals, multilingual aggregation, or automatic plotting of weighted F1; weighted F1 will not be plotted unless the toggle is manually enabled. ([GitHub][12])

[1]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/config.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/baseline/train_baseline.py "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/baseline/train_baseline_sentiment.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/find_neuron/find_neuron.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/prepare_probeless/prepare_masks.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/train_neuron/train_neft.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/train_neuron/train_probeless.py "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/train_neuron/train_neft_sentiment.py "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/train_neuron/train_probeless_sentiment.py "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/evaluation/evaluate_model.py "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/evaluation/evaluate_model_sentiment.py "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/PharasiG/NeFT/LLR/evaluation/plot.py "raw.githubusercontent.com"
