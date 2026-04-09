import torch
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import csv
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config


def calculate_model_weight_cosine(
    base_model_name,
    finetuned_model_path,
    save_dir="./temp_cosine_outputs"
):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "cos_out.csv")):
        print(f"Cosine similarity already calculated at {save_dir}")
        return

    print(f"Loading original model: {base_model_name}")
    model_1 = AutoModel.from_pretrained(base_model_name)

    print(f"Loading fine-tuned model: {finetuned_model_path}")
    model_2 = AutoModel.from_pretrained(finetuned_model_path)

    W_out_1 = []
    W_out_2 = []

    print("Extracting weights from encoder layers...")
    for layer_idx in range(len(model_1.encoder.layer)):
        w_out_1 = model_1.encoder.layer[layer_idx].output.dense.weight.data.cpu().numpy()
        w_out_2 = model_2.encoder.layer[layer_idx].output.dense.weight.data.cpu().numpy()

        W_out_1.append(w_out_1)
        W_out_2.append(w_out_2)

    print(f"W_out shape: {W_out_1[0].shape}")

    neuron_corr = {}
    print("Calculating cosine similarities for out projection...")

    for layer_idx, (w1_layer, w2_layer) in enumerate(zip(W_out_1, W_out_2)):
        num_neurons = w1_layer.shape[0]

        for neuron_idx in range(num_neurons):
            neuron_1 = w1_layer[neuron_idx]
            neuron_2 = w2_layer[neuron_idx]

            score = cosine_similarity([neuron_1], [neuron_2])
            neuron_corr[(layer_idx, neuron_idx)] = score[0][0]

    corr_df = pd.DataFrame({"corr": pd.Series(neuron_corr)})
    corr_df.index.names = ["layer_idx", "neuron_idx"]
    corr_df = corr_df.reset_index()
    corr_df["abs_corr"] = np.abs(corr_df["corr"].values)

    corr_df = corr_df.sort_values(by="corr", ascending=True)
    corr_df = corr_df.reset_index(drop=True)

    output_path = os.path.join(save_dir, "cos_out.csv")
    corr_df.to_csv(output_path, index=False)
    print(f"Saved out correlations to {output_path}")


def get_threshold_score(csv_path, neuron_number=10000):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = csv.reader(f)
        score_list = []

        for idx, row in enumerate(rows):
            if idx == 0:
                continue
            score_list.append(float(row[2]))

    score_list.sort()

    if neuron_number > len(score_list):
        neuron_number = len(score_list)

    threshold = score_list[neuron_number - 1] if neuron_number > 0 else score_list[0]
    return threshold


def get_neuron_dict(cosine_dir, threshold=0.9):
    cos_out_path = os.path.join(cosine_dir, "cos_out.csv")
    neuron_dict = {}

    with open(cos_out_path, "r", encoding="utf-8") as f:
        rows = csv.reader(f)
        for idx, row in enumerate(rows):
            if idx == 0:
                continue

            neuron_layer = int(row[0])
            neuron_idx = int(row[1])
            score = float(row[2])

            if score < threshold:
                layer_name = f"{neuron_layer}_out"
                if layer_name not in neuron_dict:
                    neuron_dict[layer_name] = [neuron_idx]
                else:
                    neuron_dict[layer_name].append(neuron_idx)

    return neuron_dict


def make_mask_and_save(
    neuron_dict,
    save_path,
    intermediate_size=3072,
    hidden_size=768
):
    mask_dict = {}

    for layer_idx in range(12):
        layer_name = f"{layer_idx}_out"
        mask_dict[layer_name] = torch.zeros(hidden_size * intermediate_size)

        if layer_name not in neuron_dict:
            continue

        for neuron in neuron_dict[layer_name]:
            start_indices = torch.arange(intermediate_size) * hidden_size + neuron
            mask_dict[layer_name][start_indices] = 1

    torch.save(mask_dict, save_path)
    print(f"Saved mask to {save_path}")

    return mask_dict


def run_find_neurons():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py")
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    cfg = config.EXPERIMENTS[args.lang]

    base_model_name = cfg["model_name"]
    finetuned_model_path = f"./output/{args.lang}-baseline"

    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(
            f"Baseline model not found at {finetuned_model_path}. Run train_baseline.py first."
        )

    save_dir = os.path.join(config.OUTPUT_MASK_DIR, args.lang, "baseline_neft")
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== NeFT Phase 2: Generating Baseline Masks for {args.lang} ===")
    print(f"Using base model: {base_model_name}")
    print(f"Using baseline checkpoint: {finetuned_model_path}")

    calculate_model_weight_cosine(
        base_model_name=base_model_name,
        finetuned_model_path=finetuned_model_path,
        save_dir=save_dir
    )

    percents = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    total_neurons = 12 * 768

    for p in percents:
        neuron_number = int(total_neurons * (p / 100.0))
        print(f"\n--- Processing {p}% ({neuron_number} neurons) ---")

        threshold = get_threshold_score(
            csv_path=os.path.join(save_dir, "cos_out.csv"),
            neuron_number=neuron_number
        )

        neuron_dict = get_neuron_dict(
            cosine_dir=save_dir,
            threshold=threshold
        )

        mask_path = os.path.join(save_dir, f"{p}neft_mask.pt")
        make_mask_and_save(
            neuron_dict=neuron_dict,
            save_path=mask_path
        )


if __name__ == "__main__":
    run_find_neurons()