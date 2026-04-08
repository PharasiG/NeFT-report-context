import os

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# ROOT PATHS
# Update these "." to the actual absolute paths if your folders are elsewhere.
DATASET_ROOT = "./data"   # Parent folder containing "UD_Arabic-PADT", "UD_Hindi-HDTB", etc.
RANKING_ROOT = "./ranking"   # Parent folder containing "POS_indo_aryan_bert", "dravidianLF_bert_pos", etc.
OUTPUT_MASK_DIR = "./masks" # Where generated masks will be saved

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
# Format:
# "key": {
#     "ranking_rel_path": Path to the ranking file (pickle) relative to RANKING_ROOT
#     "dataset_folder":   Name of the folder inside DATASET_ROOT
#     "conllu_prefix":    The prefix used in the .conllu filenames (e.g., 'mr_ufal')
# }

EXPERIMENTS = {
    # --- POS mBERT ---
    "hindi": {
        "ranking_rel_path": "hin_probeless_ranking_9984N.pkl",
        "dataset_folder": "UD_Hindi-HDTB",
        "conllu_prefix": "hi_hdtb"
    },
    "arabic": {
        "ranking_rel_path": "ara_probeless_ranking_9984N.pkl",
        "dataset_folder": "UD_Arabic-PADT",
        "conllu_prefix": "ar_padt"
    },
    "Chinese": {
        "ranking_rel_path": "bert_chinese_probeless_ranking.pkl",
        "dataset_folder": "UD_Chinese-GSD",
        "conllu_prefix": "zh_gsd"
    },
    "french": {
        "ranking_rel_path": "bert_french_probeless_ranking.pkl",
        "dataset_folder": "UD_French-GSD",
        "conllu_prefix": "fr_gsd"
    },
    # --- NER mBERT ---
    "arabic_ner": {
        "ranking_rel_path": "ner_ara_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_Arabic-NERD",
        "conllu_prefix": "ar_ner"
    },
    "chinese_ner": {
        "ranking_rel_path": "zh_NER_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_Chinese-NERD",
        "conllu_prefix": "zh_ner"
    },
    "french_ner": {
        "ranking_rel_path": "fr_NER_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_French_NERD",
        "conllu_prefix": "fr_ner"
    },
    "hindi_ner": {
        "ranking_rel_path": "ner_hin_mbert_probeless_ranking.pkl",
        "dataset_folder": "UD_Hindi-NERD",
        "conllu_prefix": "hi_ner"
    },
    # --- XLM-R POS Group ---
    "XLM_arabic_pos": {
        "ranking_rel_path": "pos_ara_xlm_probeless_ranking",
        "dataset_folder": "UD_Arabic-PADT",
        "conllu_prefix": "ar_padt"
    },
    "XLM_french_pos": {
        "ranking_rel_path": "pos_fra_xlm_probeless_ranking",
        "dataset_folder": "UD_French-GSD",
        "conllu_prefix": "fr_gsd"
    },
    "XLM_hindi_pos": {
        "ranking_rel_path": "pos_hin_xlm_probeless_ranking",
        "dataset_folder": "UD_Hindi-HDTB",
        "conllu_prefix": "hi_hdtb"
    },
    "XLM_chinese_pos": {
        "ranking_rel_path": "pos_zho_xlm_probeless_ranking",
        "dataset_folder": "UD_Chinese-GSD",
        "conllu_prefix": "zh_gsd"
    },

    # --- XLM-R NER Group ---
    "XLM_arabic_ner": {
        "ranking_rel_path": "ner_ara_xlm_probeless_ranking.pkl",
        "dataset_folder": "UD_Arabic-NERD",
        "conllu_prefix": "ar_ner"
    },
    "XLM_chinese_ner": {
        "ranking_rel_path": "ner_zho_xlm_probeless_ranking",
        "dataset_folder": "UD_Chinese-NERD",
        "conllu_prefix": "zh_ner"
    },
    "XLM_french_ner": {
        "ranking_rel_path": "ner_fra_xlm_probeless_ranking",
        "dataset_folder": "UD_French_NERD",
        "conllu_prefix": "fr_ner"
    },
    "XLM_hindi_ner": {
        "ranking_rel_path": "ner_hin_xlm_probeless_ranking.pkl",
        "dataset_folder": "UD_Hindi-NERD",
        "conllu_prefix": "hi_ner"
    },
}
