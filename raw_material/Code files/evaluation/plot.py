import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION: EDIT ONLY THIS SECTION
# =========================================================

# Input CSV path
# Option 1: direct file path

# Toggle plots
PLOT_ACCURACY = True
PLOT_PRECISION = True
PLOT_RECALL = True
PLOT_F1_MACRO = True
PLOT_F1_WEIGHTED = False

# Add any extra metric columns here if needed
EXTRA_METRICS = {
    # "MCC": False,
    # "Specificity": False,
    # "Loss": False,
}

SHOW_PLOTS = False
SAVE_PLOTS = True
FIG_SIZE = (10, 6)

# Optional title prefix
USE_LANGUAGE_IN_TITLE = True

# =========================================================
# END CONFIGURATION
# =========================================================


def normalize_model_name(name):
    name = str(name).strip().lower()
    if name == "baseline":
        return "baseline"
    if name == "neft":
        return "neft"
    if name in ["probeless", "probe-less"]:
        return "probeless"
    return name


def prettify_lang(lang):
    return str(lang).replace("_", " ").replace("-", " ").title()


def build_metric_toggle_map():
    metric_map = {
        "Accuracy": PLOT_ACCURACY,
        "Precision": PLOT_PRECISION,
        "Recall": PLOT_RECALL,
        "F1_Macro": PLOT_F1_MACRO,
        "F1_Weighted": PLOT_F1_WEIGHTED,
    }

    for metric_name, enabled in EXTRA_METRICS.items():
        metric_map[metric_name] = enabled

    return metric_map


def get_metric_output_name(metric):
    return metric.lower().replace(" ", "_")


def plot_metric(df, metric, output_dir, lang):
    if metric not in df.columns:
        print(f"[WARN] Column '{metric}' not found in CSV. Skipping.")
        return

    baseline_df = df[df["Model_Type"] == "baseline"].sort_values("Percent")
    neft_df = df[df["Model_Type"] == "neft"].sort_values("Percent")
    probeless_df = df[df["Model_Type"] == "probeless"].sort_values("Percent")

    plt.figure(figsize=FIG_SIZE)

    if not probeless_df.empty:
        plt.plot(
            probeless_df["Percent"],
            probeless_df[metric],
            marker="o",
            label="Probeless"
        )

    if not neft_df.empty:
        plt.plot(
            neft_df["Percent"],
            neft_df[metric],
            marker="s",
            label="NEFT"
        )

    if not baseline_df.empty:
        baseline_value = baseline_df[metric].iloc[0]
        plt.axhline(
            y=baseline_value,
            linestyle="--",
            label=f"Baseline ({baseline_value:.4f})"
        )

    lang_text = prettify_lang(lang)

    if USE_LANGUAGE_IN_TITLE:
        plt.title(f"{lang_text} - {metric}: Baseline vs NEFT vs Probeless")
    else:
        plt.title(f"{metric}: Baseline vs NEFT vs Probeless")

    plt.xlabel("Percent (%)")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.4)
    plt.legend()

    if SAVE_PLOTS:
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{lang}_{get_metric_output_name(metric)}_comparison.png"
        out_path = os.path.join(output_dir, file_name)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {out_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()

    LANG = args.lang
    INPUT_CSV = f"./Results/scores/evaluation_summary_{LANG}.csv"
    OUTPUT_DIR = f"./Results/metric_plots/{LANG}"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required_columns = {"Model_Type", "Percent"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    df["Model_Type"] = df["Model_Type"].apply(normalize_model_name)

    metric_toggle_map = build_metric_toggle_map()
    metrics_to_plot = [metric for metric, enabled in metric_toggle_map.items() if enabled]

    print(f"Language: {LANG}")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Metrics selected for plotting:")
    for metric in metrics_to_plot:
        print(f" - {metric}")

    for metric in metrics_to_plot:
        plot_metric(df, metric, OUTPUT_DIR, LANG)


if __name__ == "__main__":
    main()