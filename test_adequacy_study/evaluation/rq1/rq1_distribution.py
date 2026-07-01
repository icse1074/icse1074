from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from core import CRITERIA, METRICS

logger = logging.getLogger(__name__)


def plot_model_boxplots(model_result: dict, output_dir: Path, labels: dict) -> None:
    """
    Create boxplots for a single model/variation.
    """
    data = model_result["data"]
    fault_model = model_result["fault_model"]
    variation = model_result["variation"]

    colors = plt.cm.Set2(np.linspace(0, 1, len(CRITERIA)))
    criterion_labels = [labels.get(c, c) for c in CRITERIA]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS):
        plot_data = [data[c][metric] for c in CRITERIA]
        bp = ax.boxplot(
            plot_data, tick_labels=criterion_labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, (d, color) in enumerate(zip(plot_data, colors), start=1):
            jitter = np.random.uniform(-0.12, 0.12, size=len(d))
            ax.scatter(
                np.full(len(d), i) + jitter, d,
                color=color, alpha=0.5, s=14, zorder=3
            )

        ax.set_title(metric.capitalize(), fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(f"{fault_model} — {variation}", fontsize=12)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"boxplots_{fault_model.replace('/', '_')}_{variation}.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight", format="pdf")
    plt.close(fig)
    logger.info("Saved RQ1 boxplots for %s (%s) to %s", fault_model, variation, path)


def plot_all_variations(
        all_model_results: list[dict],
        output_dir: Path,
        labels: dict
) -> None:
    """
    Generate boxplots for all models and variations.

    Args:
        all_model_results: list of dicts with OR/US/MERGED keys
        output_dir: directory to save PDFs
        labels: dict mapping criterion names to display labels
    """
    for mr_dict in all_model_results:
        for variation, mr in mr_dict.items():
            plot_model_boxplots(mr, output_dir, labels)