import os
import pickle
import pandas as pd
import torch
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config # Imports your config.py

NEURONS_PER_LAYER = 768
TOTAL_NEURONS = 9216  # 12 layers * 768 neurons
PERCENTS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def generate_masks(lang_key):
    if lang_key not in config.EXPERIMENTS:
        raise ValueError(f"Language '{lang_key}' not found in config.py")

    cfg = config.EXPERIMENTS[lang_key]
    
    # 1. Setup Paths
    ranking_path = os.path.join(config.RANKING_ROOT, cfg["ranking_rel_path"])
    save_dir = os.path.join(config.OUTPUT_MASK_DIR, lang_key)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing {lang_key}...")
    print(f"  - Input Ranking: {ranking_path}")
    print(f"  - Output Dir:    {save_dir}")

    # 2. Load Pickle (pic_to_csv.py logic)
    with open(ranking_path, "rb") as f:
        data = pickle.load(f)

    # 3. Process Data (format.py & format2.py logic)
    df = pd.DataFrame(data, columns=["value"])
    df["layer_idx"] = df["value"] // NEURONS_PER_LAYER
    df["neuron_idx"] = df["value"] % NEURONS_PER_LAYER
    
    # Filter layer 0 and shift layers (format2.py)
    df = df[df["layer_idx"] != 0].copy()
    df["layer_idx"] = df["layer_idx"] - 1

    # 4. Loop Percentages (format3.py -> create_mask logic)
    for p in PERCENTS:
        # Get top Y rows
        y = int(TOTAL_NEURONS * (p / 100.0))
        subset_df = df.head(y)
        
        # Identify neurons to keep/mask
        grouped = subset_df.groupby('layer_idx')['neuron_idx'].apply(set).to_dict()
        
        # Create the mask tensor
        intermediate_size = 3072
        hidden_size = 768
        mask_dict = {}

        for layer_idx in range(12): 
            layer_name = f"{layer_idx}_out"
            mask_dict[layer_name] = torch.zeros(hidden_size * intermediate_size)

            if layer_idx in grouped:
                for neuron in grouped[layer_idx]:
                    start_idx = neuron * 3072 
                    for i in range(intermediate_size):
                        index = i * hidden_size + neuron
                        mask_dict[layer_name][index] = 1

        # Save
        save_path = os.path.join(save_dir, f"{p}probeless_mask.pt")
        torch.save(mask_dict, save_path)
        print(f"    - Saved {p}% mask")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py (e.g., marathi)")
    args = parser.parse_args()
    generate_masks(args.lang)
