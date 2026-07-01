import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
FAULT_MODELS = [
    "gpt-5-mini",
    "gpt-4.1-mini",
    "claude-haiku-4-5",
    "meta-llama_llama-3.3-70B-Instruct",
    # "deepseek-v4-flash",
]

MODEL_LABELS = {
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "claude-haiku-4-5": "Claude 4.5 Haiku",
    "meta-llama_llama-3.3-70B-Instruct": "Llama 3.3 (70B)",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
}

METRIC_TYPE = "trigger" # trigger or detection

METRICS = {
    f"full_line_coverage_{METRIC_TYPE}": "Line",
    f"full_branch_coverage_{METRIC_TYPE}": "Branch",
    f"full_mutation_{METRIC_TYPE}": "Mutation",
}

# --- Black & White Academic Style Configuration ---
# Use shades of gray for visual distinction while remaining monochrome
METRIC_COLORS = {
    f"full_line_coverage_{METRIC_TYPE}": "0.4",  # Dark Gray
    f"full_branch_coverage_{METRIC_TYPE}": "0.7",  # Medium Gray
    f"full_mutation_{METRIC_TYPE}": "1.0",  # White (will be outlined in black)
}

# Use distinct hatch patterns as primary visual distinctness
METRIC_HATCHES = {
    f"full_line_coverage_{METRIC_TYPE}": "///",  # Diagonal dense lines
    f"full_branch_coverage_{METRIC_TYPE}": "\\\\",  # Backslash medium dense
    f"full_mutation_{METRIC_TYPE}": "...",  # Dots
}


def load_data(results_folder, benchmark, guidance):
    """Iterates through files, extracts the last value of each run array,

    averages them per line, and aggregates them for the boxplot.
    """
    results_folder = Path(results_folder)
    all_data = []

    for model in FAULT_MODELS:
        file_path = (
                results_folder / model / benchmark / guidance / "minimisation_results.jsonl"
        )

        if not file_path.exists():
            print(f"Warning: File not found -> {file_path}")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data_row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip metadata/malformed lines safely

                    # Process each metric
                    for metric_key in METRICS.keys():
                        if metric_key in data_row:
                            runs = data_row[metric_key]

                            # Ensure it's a list of runs (list of lists)
                            if isinstance(runs, list):
                                final_values = []

                                for run in runs:
                                    # Mimic: [float(run[-1]) for run in runs if len(run) > 0]
                                    if isinstance(run, list) and len(run) > 0:
                                        final_values.append(float(run[-1]))
                                    elif isinstance(run, (int, float)):
                                        # Fallback just in case a run is a scalar instead of a list
                                        final_values.append(float(run))

                                # Mimic: data[criterion][metric].append(float(np.mean(final_values)))
                                if final_values:
                                    mean_value = float(np.mean(final_values))
                                    all_data.append(
                                        {
                                            "Model": model,
                                            "Metric": metric_key,
                                            "Value": mean_value,
                                        }
                                    )

        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return pd.DataFrame(all_data)


def plot_grouped_boxplot(df, title: str, output_filename: str):
    """Generates a grouped boxplot where models are on the X-axis and metrics are grouped."""
    if df.empty:
        print("No data available to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Positions setup
    num_models = len(FAULT_MODELS)
    num_metrics = len(METRICS)

    box_width = 0.2
    group_spacing = 1.0  # Distance between centers of model groups
    metric_keys = list(METRICS.keys())

    # Calculate offsets so metrics are centered around the model's tick mark
    # e.g., for 3 metrics: [-0.2, 0.0, 0.2]
    offsets = np.linspace(
        -(num_metrics - 1) * box_width / 2,
        (num_metrics - 1) * box_width / 2,
        num_metrics,
    )

    # Store legend handles
    legend_handles = {}

    # Iterate through each model and its respective metrics to plot boxes
    for model_idx, model in enumerate(FAULT_MODELS):
        model_df = df[df["Model"] == model]

        for metric_idx, metric_key in enumerate(metric_keys):
            metric_df = model_df[model_df["Metric"] == metric_key]

            if metric_df.empty:
                continue

            # Calculate exact X position for this specific box
            position = (model_idx * group_spacing) + offsets[metric_idx]

            # Plot the single boxplot
            box = ax.boxplot(
                metric_df["Value"],
                positions=[position],
                widths=box_width,
                patch_artist=True,  # Allows custom coloring
                showmeans=True,
                meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "black",
                },
            )

            # Customize the appearance (Colors & Hatches)
            rect = box["boxes"][0]
            rect.set_facecolor(METRIC_COLORS[metric_key])
            rect.set_hatch(METRIC_HATCHES[metric_key])
            rect.set_alpha(0.8)

            # Style whiskers, caps, and medians for clean aesthetics
            for element in ["whiskers", "caps"]:
                plt.setp(box[element], color="#333333", linestyle="--")
            plt.setp(box["medians"], color="black", linewidth=1.5)

            # Grab one handle per metric for the legend
            if metric_key not in legend_handles:
                legend_handles[metric_key] = rect

    # --- Formatting Plot Aesthetics ---
    # ax.set_ylabel("Values", fontsize=12, fontweight="bold")
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Set X-axis tick positions and labels
    tick_positions = np.arange(num_models) * group_spacing
    ax.set_xticks(tick_positions)

    clean_model_labels = [MODEL_LABELS.get(model, model) for model in FAULT_MODELS]

    # Apply the clean labels to the X-axis
    ax.set_xticklabels(clean_model_labels, rotation=0, ha="center", fontsize=10)
    # ax.set_xticklabels(FAULT_MODELS, rotation=15, ha="right", fontsize=10)

    # Add grid lines for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)  # Put grid lines behind plots

    # Create the legend using the formal labels
    labels = [METRICS[k] for k in legend_handles.keys()]
    ax.legend(
        legend_handles.values(),
        labels,
        title="Metrics",
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()

    # Show or save the plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


# --- Execution Block ---
if __name__ == "__main__":
    # Replace these placeholder strings with your actual arguments/variables
    RESULTS_FOLDER_PATH = "test_adequacy_study/evaluation/original_prompt/results/below_05"
    GIVEN_BENCHMARK = "mbpp"
    GIVEN_GUIDANCE = "none"
    output_filename = "output/plots/" + METRIC_TYPE + "_" + GIVEN_BENCHMARK + "_" + GIVEN_GUIDANCE + ".pdf"
    title = f"Fault {METRIC_TYPE.title()} ({GIVEN_BENCHMARK.upper()})"

    # 1. Fetch and process data
    data_df = load_data(
        results_folder=RESULTS_FOLDER_PATH,
        benchmark=GIVEN_BENCHMARK,
        guidance=GIVEN_GUIDANCE
    )

    # 2. Generate the plot
    plot_grouped_boxplot(data_df, title, output_filename)
