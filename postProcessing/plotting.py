import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_selected_runs(save_dir, runs, output_dir=None, save_individual=True):
    """
    Loads training_history.csv, roc_curve.csv, and summary.json for selected runs.

    Args:
        save_dir (str): Base directory containing run folders.
        runs (dict | list): Either a dict mapping `{run_folder_name: "Custom Legend Label"}`
                            or a list of `run_folder_name` strings.
        output_dir (str, optional): Where to save comparison plots.
        save_individual (bool): Whether to save individual summary plots in each run folder.
    """
    if output_dir is None:
        output_dir = os.path.join(save_dir, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Normalize input into a {run_id: display_label} dictionary
    run_map = runs if isinstance(runs, dict) else {r: r for r in runs}

    runs_data = []
    for run_id, display_label in run_map.items():
        run_dir = os.path.join(save_dir, run_id)
        summary_path = os.path.join(run_dir, "summary.json")

        if not os.path.exists(summary_path):
            print(f"[Warning] Skipping '{run_id}': summary.json not found in {run_dir}")
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        hist_path = os.path.join(run_dir, "training_history.csv")
        roc_path = os.path.join(run_dir, "roc_curve.csv")

        hist_df = pd.read_csv(hist_path) if os.path.exists(hist_path) else None
        roc_df = pd.read_csv(roc_path) if os.path.exists(roc_path) else None

        runs_data.append({
            "run_dir": run_dir,
            "run_id": run_id,
            "label": display_label,
            "summary": summary,
            "history": hist_df,
            "roc": roc_df,
        })

    if not runs_data:
        print("No valid runs loaded. Check your run folder names and save_dir.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs_data), 1)))

    # =========================================================================
    # 1. COMBINED LOSS CURVES (Train & Validation Loss)
    # =========================================================================
    fig_loss, ax_loss = plt.subplots(figsize=(9, 5.5))
    has_loss = False

    for idx, run in enumerate(runs_data):
        df = run["history"]
        if df is None:
            continue
        c = colors[idx % len(colors)]
        label = run["label"]

        if "loss" in df.columns:
            ax_loss.plot(df["epoch"], df["loss"], color=c, linestyle="--", alpha=0.65, label=f"{label} (train)")
            has_loss = True
        if "val_loss" in df.columns:
            ax_loss.plot(df["epoch"], df["val_loss"], color=c, linestyle="-", lw=2, label=f"{label} (val)")
            has_loss = True

    if has_loss:
        ax_loss.set_xlabel("Epoch", fontsize=11)
        ax_loss.set_ylabel("Loss", fontsize=11)
        ax_loss.set_title("Training & Validation Loss Comparison", fontsize=13, fontweight="bold")
        ax_loss.grid(alpha=0.3)
        ax_loss.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
        fig_loss.tight_layout()
        fig_loss.savefig(os.path.join(output_dir, "selected_runs_loss.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_loss)

    # =========================================================================
    # 2. COMBINED ROC CURVES & BACKGROUND REJECTION (1/FPR vs TPR)
    # =========================================================================
    fig_roc, (ax_roc, ax_rej) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for idx, run in enumerate(runs_data):
        roc_df = run["roc"]
        if roc_df is None:
            continue
        c = colors[idx % len(colors)]
        label = run["label"]
        auc_val = run["summary"].get("roc_auc", np.nan)

        fpr = roc_df["fpr"].to_numpy()
        tpr = roc_df["tpr"].to_numpy()

        ax_roc.plot(fpr, tpr, color=c, lw=2, label=f"{label} (AUC = {auc_val:.4f})")

        valid = (fpr > 0) & (tpr > 0.1)
        ax_rej.plot(tpr[valid], 1.0 / fpr[valid], color=c, lw=2, label=label)

    ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random (AUC = 0.5000)")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel("False Positive Rate (Background Efficiency)", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Signal Efficiency)", fontsize=11)
    ax_roc.set_title("ROC Curve Comparison", fontsize=12, fontweight="bold")
    ax_roc.grid(alpha=0.3)
    ax_roc.legend(loc="lower right", fontsize=9)

    ax_rej.set_yscale("log")
    ax_rej.set_xlim([0.1, 1.0])
    ax_rej.set_xlabel("Signal Efficiency (TPR)", fontsize=11)
    ax_rej.set_ylabel("Background Rejection (1 / FPR)", fontsize=11)
    ax_rej.set_title("Background Rejection Comparison", fontsize=12, fontweight="bold")
    ax_rej.grid(True, which="both", alpha=0.3)
    ax_rej.legend(loc="upper right", fontsize=9)

    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(output_dir, "selected_runs_roc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    # =========================================================================
    # 3. INDIVIDUAL RUN PLOTS (Saved inside each selected run's folder)
    # =========================================================================
    if save_individual:
        for run in runs_data:
            label = run["label"]
            hist_df = run["history"]
            roc_df = run["roc"]
            summary = run["summary"]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

            ax1 = axes[0]
            if hist_df is not None:
                metric_cols = [col for col in hist_df.columns if col != "epoch"]
                for col in metric_cols:
                    style = "--" if col.startswith("val_") else "-"
                    ax1.plot(hist_df["epoch"], hist_df[col], linestyle=style, lw=1.8, label=col)
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Value")
                ax1.set_title("Training History")
                ax1.grid(alpha=0.3)
                ax1.legend()

            ax2 = axes[1]
            if roc_df is not None:
                fpr = roc_df["fpr"].to_numpy()
                tpr = roc_df["tpr"].to_numpy()
                auc_val = summary.get("roc_auc", 0.0)
                rej_50 = summary.get("bg_rejection_at_50pct_eff", 0.0)

                ax2.plot(fpr, tpr, lw=2, color="#1f77b4", label=f"AUC = {auc_val:.4f}")
                ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

                idx_50 = np.argmin(np.abs(tpr - 0.5))
                ax2.scatter(
                    [fpr[idx_50]], [tpr[idx_50]], color="#d62728", zorder=5,
                    label=f"50% TPR (1/FPR = {rej_50:.1f})"
                )

                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curve")
                ax2.grid(alpha=0.3)
                ax2.legend(loc="lower right")

            fig.suptitle(f"Run Summary: {label}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(run["run_dir"], "run_summary_plots.png"), dpi=200)
            plt.close(fig)

    print(f"Plotted {len(runs_data)} runs. Figures saved to: {output_dir}/")


if __name__ == "__main__":
    # Map each run folder name to the exact legend label you want displayed
    runs_to_plot = {
        "w3_l2_br_ring_all": "3 Modes, 2 Layers (Ring, Homodyne)"
    }

    plot_selected_runs(save_dir="results", runs=runs_to_plot)