"""
experiments/run_doe_comparison.py
==================================
Main entry point for the DOE vs. Random Initialization comparison.

WHAT THIS SCRIPT DOES
----------------------
Runs the active learning loop under THREE initialization conditions:

  Condition A – Random Init     (the baseline from your teammate's work)
  Condition B – MaxMin DOE      (greedy Tanimoto diversity; DOE analog)
  Condition C – k-Medoids DOE   (cluster-based; LHS analog)

For each condition:
  - Runs across N_SEEDS seeds (default 3, matching your teammate)
  - Saves per-seed results to Drive
  - At the end, computes mean ± std across seeds for plotting

All three conditions use:
  - Same base learner (Random Forest, consistent with your teammate)
  - Same batch size (500, from your teammate's grid search)
  - Same init fraction (20%, from your teammate's starting-fraction experiment)
  - Same uncertainty sampling acquisition function (RF entropy)
  - Same scaffold split test set

This isolates the effect of initialization strategy.

HOW TO RUN FROM COLAB
----------------------
    from google.colab import drive
    drive.mount('/content/drive')
    import sys
    sys.path.insert(0, '/content/drive/MyDrive/AutomationS26/HW3/project/')
    %run experiments/run_doe_comparison.py

OR as a CLI command:
    !python experiments/run_doe_comparison.py

OUTPUTS (saved to results/ directory)
---------------------------------------
  results/doe_comparison_seed{seed}.json    – per-seed raw results
  results/doe_comparison_summary.json       – mean ± std across seeds
  results/doe_comparison_plot.png           – the main comparison figure
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ── make sure project root is on the path ─────────────────────────────────────
# When running from Colab as !python experiments/..., the CWD may vary.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.data.data_loader import load_hiv_data
from initialization.random_init  import random_initialization
from initialization.maxmin_init  import maxmin_initialization
from initialization.kmedoids_init import kmedoids_initialization
from models.random_forest_model  import RandomForestModel
from active_learning.al_loop     import run_active_learning
from evaluation.metrics          import EvalResult


# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  –  change these values to reproduce different experiments
# ════════════════════════════════════════════════════════════════════════════

INIT_FRACTION = 0.20        # 20% starting pool (your teammate's finding)
BATCH_SIZE    = 500         # from your teammate's batch-size grid search
N_SEEDS       = 3           # seeds to average over (matching existing work)
RF_ESTIMATORS = 100         # number of trees in RF
RF_MAX_DEPTH  = None        # None = grow fully (adjust if overfitting)
RESULTS_DIR   = os.path.join(_ROOT, "results")

# ════════════════════════════════════════════════════════════════════════════


def results_to_dict(results):
    """Convert list[EvalResult] to a JSON-serialisable dict."""
    return {
        "n_labeled": [r.n_labeled for r in results],
        "auprc":     [r.auprc     for r in results],
        "auroc":     [r.auroc     for r in results],
        "hit_recovery": [r.hit_recovery for r in results],
    }


def run_one_condition(
    condition_name: str,
    init_fn,           # callable returning np.ndarray of selected indices
    data,              # DataBundle
    seed: int,
) -> dict:
    """
    Run one (condition, seed) combination and return serialisable results.

    Parameters
    ----------
    condition_name : str   – label for logging ("Random", "MaxMin", "kMedoids")
    init_fn        : callable(seed) -> np.ndarray
    data           : DataBundle from load_hiv_data()
    seed           : int
    """
    n_total = len(data.y_train_pool)
    n_init  = int(n_total * INIT_FRACTION)

    print(f"\n{'='*60}")
    print(f"  Condition : {condition_name}")
    print(f"  Seed      : {seed}")
    print(f"  n_init    : {n_init:,} / {n_total:,} ({100*INIT_FRACTION:.0f}%)")
    print(f"{'='*60}")

    t0 = time.time()

    # ── 1. Select initial labelled set ───────────────────────────────────────
    init_indices = init_fn(seed=seed)
    print(f"  Initialisation done in {time.time()-t0:.1f}s")

    # ── 2. Build (untrained) model ────────────────────────────────────────────
    model = RandomForestModel(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        seed=seed,
    )

    # ── 3. Run active learning loop ───────────────────────────────────────────
    results = run_active_learning(
        model        = model,
        X_pool       = data.X_train_pool,
        y_pool       = data.y_train_pool,
        X_test       = data.X_test,
        y_test       = data.y_test,
        init_indices = init_indices,
        batch_size   = BATCH_SIZE,
        seed         = seed,
        verbose      = True,
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed/60:.1f} min")

    return results_to_dict(results)


def aggregate_seeds(all_seed_results: list) -> dict:
    """
    Given list of per-seed result dicts, compute mean ± std across seeds.

    We interpolate each run onto a common n_labeled grid so that different
    runs (which may have slightly different iteration counts) are comparable.
    """
    # Find the common x-axis grid (use the run with the most iterations)
    max_len = max(len(r["n_labeled"]) for r in all_seed_results)
    ref_x   = all_seed_results[0]["n_labeled"]   # use first seed as x grid

    auprc_arrays = []
    auroc_arrays = []

    for r in all_seed_results:
        x = np.array(r["n_labeled"])
        # Interpolate onto ref_x grid
        auprc_interp = np.interp(ref_x, x, r["auprc"])
        auroc_interp = np.interp(ref_x, x, r["auroc"])
        auprc_arrays.append(auprc_interp)
        auroc_arrays.append(auroc_interp)

    auprc_mat = np.stack(auprc_arrays)   # (n_seeds, n_iters)
    auroc_mat = np.stack(auroc_arrays)

    return {
        "n_labeled":   ref_x,
        "auprc_mean":  auprc_mat.mean(axis=0).tolist(),
        "auprc_std":   auprc_mat.std(axis=0).tolist(),
        "auroc_mean":  auroc_mat.mean(axis=0).tolist(),
        "auroc_std":   auroc_mat.std(axis=0).tolist(),
    }


def plot_doe_comparison(summary: dict, save_path: str):
    """
    Reproduce the style of Figure 7/8 in your milestone update.

    Three lines (Random, MaxMin, kMedoids), each with ±1 std shading.
    Primary metric: AUPRC (as in your teammate's plots).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = {
        "Random":   "#1f77b4",   # blue
        "MaxMin":   "#2ca02c",   # green
        "kMedoids": "#ff7f0e",   # orange
    }

    for metric, ax, title in [
        ("auprc", axes[0], "AUPRC (Average Precision) – Test Set"),
        ("auroc", axes[1], "AUROC – Test Set"),
    ]:
        for cond_name, cond_data in summary.items():
            x     = np.array(cond_data["n_labeled"])
            mean  = np.array(cond_data[f"{metric}_mean"])
            std   = np.array(cond_data[f"{metric}_std"])
            colour = colours.get(cond_name, "grey")

            ax.plot(x, mean, label=cond_name, color=colour, linewidth=2)
            ax.fill_between(x, mean - std, mean + std,
                            color=colour, alpha=0.15)

        ax.set_xlabel("Number of Labeled Molecules in Training Set", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add horizontal dashed line at random-classifier baseline
        if metric == "auprc":
            ax.axhline(0.035, linestyle="--", color="grey",
                       alpha=0.6, label="Random baseline (≈3.5%)")

    plt.suptitle(
        f"DOE vs. Random Initialization Comparison\n"
        f"(RF, uncertainty sampling, batch={BATCH_SIZE}, "
        f"init={int(100*INIT_FRACTION)}%, {N_SEEDS} seeds)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data (cached after first run) ────────────────────────────────────
    data = load_hiv_data()
    n_total = len(data.y_train_pool)
    n_init  = int(n_total * INIT_FRACTION)

    # ── Define the three initialisation strategies ────────────────────────────
    #
    # Each entry is  (condition_name,  init_function_taking_seed)
    #
    # We use lambda to partially-apply the fingerprints and n_init so that
    # run_one_condition only needs to call  init_fn(seed=seed).

    conditions = [
        (
            "Random",
            lambda seed: random_initialization(
                n_total=n_total,
                n_init=n_init,
                seed=seed,
            ),
        ),
        (
            "MaxMin",
            lambda seed: maxmin_initialization(
                X_pool=data.X_train_pool,
                n_init=n_init,
                seed=seed,
                verbose=True,
            ),
        ),
        (
            "kMedoids",
            lambda seed: kmedoids_initialization(
                X_pool=data.X_train_pool,
                n_init=n_init,
                seed=seed,
                subsample_size=n_init + 500,  # always slightly larger than n_init
                verbose=True,
            ),
        ),
    ]

    # ── Run all conditions across all seeds ───────────────────────────────────
    all_results = {cond_name: [] for cond_name, _ in conditions}

    for cond_name, init_fn in conditions:
        for seed in range(N_SEEDS):
            result = run_one_condition(
                condition_name=cond_name,
                init_fn=init_fn,
                data=data,
                seed=seed,
            )
            all_results[cond_name].append(result)

            # Save checkpoint after each (condition, seed) so a Colab crash
            # doesn't lose everything
            ckpt_path = os.path.join(
                RESULTS_DIR,
                f"doe_comparison_{cond_name.lower()}_seed{seed}.json"
            )
            with open(ckpt_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Checkpoint saved: {ckpt_path}")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    summary = {}
    for cond_name, seed_results in all_results.items():
        summary[cond_name] = aggregate_seeds(seed_results)

    summary_path = os.path.join(RESULTS_DIR, "doe_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = os.path.join(RESULTS_DIR, "doe_comparison_plot.png")
    plot_doe_comparison(summary, save_path=plot_path)


if __name__ == "__main__":
    main()
