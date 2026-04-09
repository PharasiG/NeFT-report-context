"""
Shared data utilities for sentiment analysis experiments.

Provides a single deterministic 3-way split (train / val / test)
so that every script in the pipeline sees exactly the same partitions.

Split ratios:  70% train  |  15% validation  |  15% test
Seeds are fixed so the split is perfectly reproducible.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# These must never change once experiments begin
_SPLIT_SEED = 42
_TEST_RATIO = 0.15      # 15% of full data
_VAL_RATIO = 0.15       # 15% of full data  (→ train gets the remaining 70%)


def load_sentiment_splits(cfg, dataset_root):
    """
    Load a sentiment CSV and return (train_df, val_df, test_df).

    The split is stratified by label and fully deterministic.
    Every script that calls this function with the same CSV will
    get identical partitions.

    Args:
        cfg:          experiment dict from config.py (needs 'dataset_folder' and 'data_file')
        dataset_root: config.DATASET_ROOT path

    Returns:
        train_df, val_df, test_df  — each a pandas DataFrame with ['text', 'label']
    """
    csv_path = os.path.join(dataset_root, cfg["dataset_folder"], cfg["data_file"])
    df = pd.read_csv(csv_path, encoding="utf-8").dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    # Step 1: carve off the test set  (15 %)
    remaining_df, test_df = train_test_split(
        df,
        test_size=_TEST_RATIO,
        random_state=_SPLIT_SEED,
        stratify=df["label"],
    )

    # Step 2: split remainder into train and val
    # val should be 15% of the *original* data → 15/85 ≈ 0.1765 of the remainder
    val_frac = _VAL_RATIO / (1 - _TEST_RATIO)
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=val_frac,
        random_state=_SPLIT_SEED,
        stratify=remaining_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df
