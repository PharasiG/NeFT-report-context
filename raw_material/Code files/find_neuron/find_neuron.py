import torch
from transformers import AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import csv
import json
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config  # Import config.py

def cal_model_weight_cos_bert(
    model_path_1='xlm-roberta-base',
    model_path_2=None,
    save_dir='neft_results/mbert_imdb/Out'
):
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Optimization: Skip calculation if CSV already exists
    if os.path.exists(os.path.join(save_dir, 'cos_out.csv')):
        print(f"Cosine similarity already calculated at {save_dir}")
        return

    print(f"Loading original model: {model_path_1}")
    model_1 = AutoModel.from_pretrained(model_path_1)

    print(f"Loading fine-tuned model: {model_path_2}")
    model_2 = AutoModel.from_pretrained(model_path_2)

    # Extract MLP weights from BERT layers
    W_out_1 = [] 
    W_out_2 = []

    print("Extracting weights from BERT layers...")
    for layer_idx in range(len(model_1.encoder.layer)):
        w_out_1 = model_1.encoder.layer[layer_idx].output.dense.weight.data.cpu().numpy()
        w_out_2 = model_2.encoder.layer[layer_idx].output.dense.weight.data.cpu().numpy()

        W_out_1.append(w_out_1)
        W_out_2.append(w_out_2)

    print(f"W_out shape: {W_out_1[0].shape}") 

    # Calculate cosine similarities
    igo_list = ['out']

    for igo in igo_list:
        if igo == 'out':
            path_tail = '/cos_out.csv'
            W_1 = W_out_1
            W_2 = W_out_2

        neuron_corr = {}
        print(f"Calculating cosine similarities for {igo} projection...")

        for layer_idx, (w1_layer, w2_layer) in enumerate(zip(W_1, W_2)):
            num_neurons = w1_layer.shape[0]

            for neuron_idx in range(num_neurons):
                neuron_1 = w1_layer[neuron_idx]
                neuron_2 = w2_layer[neuron_idx]

                score = cosine_similarity([neuron_1], [neuron_2])
                neuron_corr[(layer_idx, neuron_idx)] = score[0][0]

        # Save to CSV
        corr_df = pd.DataFrame({'corr': pd.Series(neuron_corr)})
        corr_df.index.names = ['layer_idx', 'neuron_idx']
        corr_df = corr_df.reset_index()
        corr_df['abs_corr'] = np.abs(corr_df['corr'].values)
        
        corr_df = corr_df.sort_values(by='corr', ascending=True)
        corr_df = corr_df.reset_index(drop=True)

        output_path = save_dir + path_tail
        corr_df.to_csv(output_path, index=False)
        print(f"Saved {igo} correlations to {output_path}")

def get_threshold_score(
        path_2=None,
        neuron_number=10000):
    
    # print(f"Calculating threshold for {neuron_number} neurons...") # Reduce spam

    f_2 = open(path_2, 'r', encoding='utf-8')
    rows_2 = csv.reader(f_2)

    score_list = []

    for idx2, row in enumerate(rows_2):
        if idx2 == 0: continue
        score = float(row[2])
        score_list.append(score)

    score_list.sort()

    if neuron_number > len(score_list):
        neuron_number = len(score_list)
        
    threshold = score_list[neuron_number - 1] if neuron_number > 0 else score_list[0]
    # print(f"Threshold score for top {neuron_number} neurons: {threshold}")

    f_2.close()
    return threshold

def get_neuron(
        path_dir='neft_results/mbert_imdb',
        threshold=0.9,
        neuron_number=10000):
    
    cos_out_path = path_dir + '/cos_out.csv'
    
    # neuron_dir = os.path.join(path_dir, str(neuron_number)) 
    # os.makedirs(neuron_dir, exist_ok=True)

    count = 0
    neuron_dict = {}

    for path_idx, path in enumerate([cos_out_path]):
        f = open(path, 'r', encoding='utf-8')
        rows = csv.reader(f)
        igo = 'out'
        for idx, row in enumerate(rows):
            if idx == 0: continue

            neuron_layer = int(row[0])
            neuron_idx = int(row[1])
            score = float(row[2])

            if score < threshold:
                count += 1
                layer_name = f"{neuron_layer}_{igo}"
                if layer_name not in neuron_dict:
                    neuron_dict[layer_name] = [neuron_idx]
                else:
                    neuron_dict[layer_name].append(neuron_idx)
        f.close()

    return neuron_dict

def make_mask_with_dict_and_save_bert_neuron(
        neuron_dict=None, # Changed to accept dict directly
        save_path=None,
        intermediate_size=3072,
        hidden_size=768
):
    mask_dict = {}

    for layer_idx in range(12):
        for igo in ['out']:
            layer_name = f"{layer_idx}_{igo}"
            mask_dict[layer_name] = torch.zeros(hidden_size * intermediate_size)

            if layer_name not in neuron_dict:
                continue

            for neuron in neuron_dict[layer_name]:
                # Optimized vector assignment to speed up the loop
                start_indices = torch.arange(intermediate_size) * hidden_size + neuron
                mask_dict[layer_name][start_indices] = 1

    torch.save(mask_dict, save_path)
    print(f"Saved mask to {save_path}")

    return mask_dict

def run_get_mbert_neuron():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, help="Language key from config.py")
    args = parser.parse_args()

    if args.lang not in config.EXPERIMENTS:
        raise ValueError(f"Language '{args.lang}' not defined in config.py")

    # 2. Setup Paths
    model_path_1 = 'xlm-roberta-base'
    # Path to the baseline model you trained with train_baseline.py
    model_path_2 = f"./output/{args.lang}-baseline"
    
    if not os.path.exists(model_path_2):
         raise FileNotFoundError(f"Baseline model not found at {model_path_2}. Run train_baseline.py first.")

    # Save outputs to ./masks/{lang}/baseline_derived/
    save_dir = os.path.join(config.OUTPUT_MASK_DIR, args.lang, "baseline_neft")
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== NeFT Phase 2: Generating Baseline Masks for {args.lang} ===")
    
    # 3. Calculate Similarity (Once)
    cal_model_weight_cos_bert(
        model_path_1=model_path_1,
        model_path_2=model_path_2,
        save_dir=save_dir
    )

    # 4. Loop through Percentages
    percents = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    total_neurons = 12 * 768

    for p in percents:
        neuron_number = int(total_neurons * (p / 100.0))
        print(f"\n--- Processing {p}% ({neuron_number} neurons) ---")

        # Get Threshold
        threshold = get_threshold_score(
            path_2=os.path.join(save_dir, 'cos_out.csv'),
            neuron_number=neuron_number
        )

        # Get Neurons
        neuron_dict = get_neuron(
            path_dir=save_dir,
            threshold=threshold,
            neuron_number=neuron_number
        )

        # Generate Mask
        mask_path = os.path.join(save_dir, f"{p}neft_mask.pt")
        make_mask_with_dict_and_save_bert_neuron(
            neuron_dict=neuron_dict,
            save_path=mask_path
        )

if __name__ == '__main__':
    run_get_mbert_neuron()
