import os

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# ROOT PATHS
# Update these "." to the actual absolute paths if your folders are elsewhere.
DATASET_ROOT = "./data"      # Parent folder containing datasets
RANKING_ROOT = "./ranking"   # Parent folder containing ranking files
OUTPUT_MASK_DIR = "./masks"  # Where generated masks will be saved

# =============================================================================
# MODEL CONSTANTS
# =============================================================================
MBERT_MODEL_NAME = "bert-base-multilingual-cased"
XLMR_MODEL_NAME = "xlm-roberta-base"

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
# Format:
# "key": {
#     "model_name":       HF model name
#     "ranking_rel_path": Path to the ranking file relative to RANKING_ROOT
#     "dataset_folder":   Name of the folder inside DATASET_ROOT
#     "conllu_prefix":    The prefix used in the .conllu filenames
# }

EXPERIMENTS = {
    # --- POS mBERT ---
    "mBERT_hindi_pos": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "hin_probeless_ranking_9984N.pkl",
        "dataset_folder": "UD_Hindi-HDTB",
        "conllu_prefix": "hi_hdtb"
    },
    "mBERT_arabic_pos": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "ara_probeless_ranking_9984N.pkl",
        "dataset_folder": "UD_Arabic-PADT",
        "conllu_prefix": "ar_padt"
    },
    "mBERT_Chinese_pos": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "bert_chinese_probeless_ranking.pkl",
        "dataset_folder": "UD_Chinese-GSD",
        "conllu_prefix": "zh_gsd"
    },
    "mBERT_french_pos": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "bert_french_probeless_ranking.pkl",
        "dataset_folder": "UD_French-GSD",
        "conllu_prefix": "fr_gsd"
    },

    # --- NER mBERT ---
    "mBERT_arabic_ner": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "ner_ara_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_Arabic-NERD",
        "conllu_prefix": "ar_ner"
    },
    "mBERT_chinese_ner": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "zh_NER_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_Chinese-NERD",
        "conllu_prefix": "zh_ner"
    },
    "mBERT_french_ner": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "fr_NER_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_French_NERD",
        "conllu_prefix": "fr_ner"
    },
    "mBERT_hindi_ner": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "ner_hin_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_Hindi-NERD",
        "conllu_prefix": "hi_ner"
    },

    # --- XLM-R POS ---
    "XLM_arabic_pos": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "pos_ara_xlm_probeless_ranking",
        "dataset_folder": "UD_Arabic-PADT",
        "conllu_prefix": "ar_padt"
    },
    "XLM_french_pos": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "pos_fra_xlm_probeless_ranking",
        "dataset_folder": "UD_French-GSD",
        "conllu_prefix": "fr_gsd"
    },
    "XLM_hindi_pos": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "pos_hin_xlm_probeless_ranking",
        "dataset_folder": "UD_Hindi-HDTB",
        "conllu_prefix": "hi_hdtb"
    },
    "XLM_chinese_pos": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "pos_zho_xlm_probeless_ranking",
        "dataset_folder": "UD_Chinese-GSD",
        "conllu_prefix": "zh_gsd"
    },

    # --- XLM-R NER ---
    "XLM_arabic_ner": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "ner_ara_xlm_probeless_ranking.pkl",
        "dataset_folder": "UD_Arabic-NERD",
        "conllu_prefix": "ar_ner"
    },
    "XLM_chinese_ner": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "ner_zho_xlm_probeless_ranking",
        "dataset_folder": "UD_Chinese-NERD",
        "conllu_prefix": "zh_ner"
    },
    "XLM_french_ner": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "ner_fra_xlm_probeless_ranking",
        "dataset_folder": "UD_French_NERD",
        "conllu_prefix": "fr_ner"
    },
    "XLM_hindi_ner": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "ner_hin_xlm_probeless_ranking.pkl",
        "dataset_folder": "UD_Hindi-NERD",
        "conllu_prefix": "hi_ner"
    },
    
    # --- mBERT Sentiment ---
    "mBERT_arabic_sentiment": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "mBERT_arabic_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "arabic_sentiment_clean_10k.csv"
    },
    "mBERT_chinese_sentiment": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "mBERT_chinese_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "chinese_sentiment_clean_10k.csv"
    },
    "mBERT_hindi_sentiment": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "mBERT_hindi_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "cleaned_hindi_data.csv"
    },
    "mBERT_french_sentiment": {
        "model_name": MBERT_MODEL_NAME,
        "ranking_rel_path": "mBERT_french_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "french_sentiment_clean_10k.csv"
    },

    # --- XLM-R Sentiment ---
    "XLM_arabic_sentiment": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "XLM_arabic_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "arabic_sentiment_clean_10k.csv"
    },
    "XLM_chinese_sentiment": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "XLM_chinese_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "chinese_sentiment_clean_10k.csv"
    },
    "XLM_hindi_sentiment": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "XLM_hindi_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "cleaned_hindi_data.csv"
    },
    "XLM_french_sentiment": {
        "model_name": XLMR_MODEL_NAME,
        "ranking_rel_path": "XLM_french_sentiment_ranking",
        "dataset_folder": "sentiment",
        "data_file": "french_sentiment_clean_10k.csv"
    },
}