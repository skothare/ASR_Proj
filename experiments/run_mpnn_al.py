"""
experiments/run_mpnn_al.py

MPNN active learning experiment

What this experiment answers:
1. Does MPNN-in-AL-loop outperform RF-in-AL-loop for active recovery?
2. Does imbalance-aware acquisition (weighted/BALD) outperform standard
   entropy sampling under severe class imbalance?
3. Is the MPNN's MC Dropout uncertainty signal better calibrated than
   RF tree variance?

Design:
We run a 2×3 grid:
  Base learner:  RF,   MPNN
  Acquisition:   entropy,  weighted,  bald (MPNN only)

All conditions use:
  - Same scaffold split (from DataBundle)
  - Same 20% random initialization
  - Same batch size (500)
  - Same 3 seeds

How to run from Colab:
After mounting Drive and setting PROJECT_ROOT in colab_runner.ipynb:

    from experiments.run_mpnn_al import main
    main()

Or to run a quick single-seed test:

    from experiments.run_mpnn_al import run_one_condition, load_all_data
    data, graphs = load_all_data()
    # Run MPNN with entropy, 1 seed
    results = run_one_condition('MPNN', 'entropy', data, graphs, seeds=[0])
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from preprocessing.data.data_loader import load_hiv_data
from preprocessing.data.graph_builder import build_graph_dataset
from initialization.random_init import random_initialization
from models.random_forest_model import RandomForestModel
from models.mpnn_model import MPNNModel
from active_learning.al_loop import run_active_learning
from evaluation.metrics import EvalResult


#Configuration

INIT_FRACTION  = 0.20
BATCH_SIZE     = 500
N_SEEDS        = 3
RESULTS_DIR    = os.path.join(_ROOT, "results")

# MPNN hyperparameters — these match the complexity of Chemprop's defaults
MPNN_CONFIG = dict(
    hidden_dim  = 128,
    num_layers  = 3,
    dropout_p   = 0.2,
    n_epochs    = 50,     # reduce to 20 for fast debugging
    batch_size  = 32,
    lr          = 1e-3,
    pos_weight  = 27.0,   # HIV: ~27× more inactives than actives
    mc_samples  = 30,
)

RF_CONFIG = dict(
    n_estimators = 100,
    max_depth    = None,
)

# Data loading and graph building


def load_all_data():
    """
    Load HIV dataset + build graph objects.
    Graphs are expensive to build — load once and reuse across conditions.
    """
    print("Loading HIV dataset and building graphs...")
    data = load_hiv_data()

    print("Building PyG graphs for training pool...")
    graphs_train, valid_train = build_graph_dataset(
        data.smiles_train, data.y_train_pool, verbose=True
    )
    print("Building PyG graphs for test set...")
    graphs_test, valid_test = build_graph_dataset(
        data.smiles_test, data.y_test, verbose=True
    )

    # Filter pool arrays to only valid (parseable) molecules
    # In practice almost all HIV SMILES parse successfully
    X_train_valid = data.X_train_pool[valid_train]
    y_train_valid = data.y_train_pool[valid_train]
    X_test_valid  = data.X_test[valid_test]
    y_test_valid  = data.y_test[valid_test]

    print(f"\nPool: {len(graphs_train):,} molecules, "
          f"{y_train_valid.sum()} actives ({100*y_train_valid.mean():.1f}%)")
    print(f"Test: {len(graphs_test):,} molecules, "
          f"{y_test_valid.sum()} actives")

    return (X_train_valid, y_train_valid, X_test_valid, y_test_valid,
            graphs_train, graphs_test)


def results_to_dict(results):
    return {
        "n_labeled":    [r.n_labeled for r in results],
        "auprc":        [r.auprc for r in results],
        "auroc":        [r.auroc for r in results],
        "hit_recovery": [r.hit_recovery for r in results],
    }


def run_one_condition(
    learner_name: str,
    acquisition: str,
    X_pool, y_pool, X_test, y_test,
    graphs_pool, graphs_test,
    seeds=None,
):
    """
    Run one (learner, acquisition) condition across all seeds.
    Returns list of per-seed result dicts.
    """
    if seeds is None:
        seeds = list(range(N_SEEDS))

    n_total = len(y_pool)
    n_init  = int(n_total * INIT_FRACTION)

    seed_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  {learner_name} | acquisition={acquisition} | seed={seed}")
        print(f"{'='*60}")
        t0 = time.time()

        # Init: random for all MPNN experiments (DOE already tested separately)
        init_indices = random_initialization(n_total, n_init, seed=seed)

        # Build model
        if learner_name == 'RF':
            model = RandomForestModel(seed=seed, **RF_CONFIG)
        elif learner_name == 'MPNN':
            model = MPNNModel(seed=seed, **MPNN_CONFIG)
        else:
            raise ValueError(f"Unknown learner: {learner_name}")

        # Run AL loop
        results = run_active_learning(
            model        = model,
            X_pool       = X_pool,
            y_pool       = y_pool,
            X_test       = X_test,
            y_test       = y_test,
            init_indices = init_indices,
            batch_size   = BATCH_SIZE,
            seed         = seed,
            verbose      = True,
            graphs_pool  = graphs_pool if learner_name == 'MPNN' else None,
            graphs_test  = graphs_test if learner_name == 'MPNN' else None,
            acquisition  = acquisition,
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed/60:.1f} min")

        r_dict = results_to_dict(results)
        seed_results.append(r_dict)

        # Checkpoint
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cond_str = f"{learner_name.lower()}_{acquisition}_seed{seed}"
        path = os.path.join(RESULTS_DIR, f"mpnn_al_{cond_str}.json")
        with open(path, 'w') as f:
            json.dump(r_dict, f, indent=2)
        print(f"  Saved: {path}")

    return seed_results


def aggregate_seeds(seed_results):
    """Mean ± std across seeds, interpolated onto common x-axis."""
    ref_x = seed_results[0]['n_labeled']
    auprc_mat = np.stack([
        np.interp(ref_x, r['n_labeled'], r['auprc']) for r in seed_results
    ])
    auroc_mat = np.stack([
        np.interp(ref_x, r['n_labeled'], r['auroc']) for r in seed_results
    ])
    hits_mat = np.stack([
        np.interp(ref_x, r['n_labeled'],
                  [h if h is not None else 0 for h in r['hit_recovery']])
        for r in seed_results
    ])
    return {
        'n_labeled':  ref_x,
        'auprc_mean': auprc_mat.mean(0).tolist(),
        'auprc_std':  auprc_mat.std(0).tolist(),
        'auroc_mean': auroc_mat.mean(0).tolist(),
        'auroc_std':  auroc_mat.std(0).tolist(),
        'hits_mean':  hits_mat.mean(0).tolist(),
        'hits_std':   hits_mat.std(0).tolist(),
    }


def plot_results(summary: dict, save_path: str):
    """Plot all conditions on AUPRC, AUROC, and hit recovery."""
    colours = {
        'RF_entropy':    '#1f77b4',   # blue
        'RF_weighted':   '#aec7e8',   # light blue
        'MPNN_entropy':  '#2ca02c',   # green
        'MPNN_weighted': 'orange', # orange
        'MPNN_bald':     'red',# red
    }
    styles = {
        'RF_entropy':    '-',
        'RF_weighted':   '--',
        'MPNN_entropy':  '-',
        'MPNN_weighted': '--',
        'MPNN_bald':     ':',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for cond_key, agg in summary.items():
        colour = colours.get(cond_key, 'gray')
        style  = styles.get(cond_key, '-')
        x      = np.array(agg['n_labeled'])

        for metric, ax in zip(['auprc', 'auroc', 'hits'], axes):
            mean = np.array(agg[f'{metric}_mean'])
            std  = np.array(agg[f'{metric}_std'])
            ax.plot(x, mean, label=cond_key, color=colour,
                    linestyle=style, linewidth=2)
            ax.fill_between(x, mean-std, mean+std,
                           color=colour, alpha=0.1)

    titles = ['AUPRC — Test Set', 'AUROC — Test Set', 'Hit Recovery Rate']
    ylabels = ['Average Precision', 'AUROC', 'Fraction of actives found']

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel('Labeled molecules in training set', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].axhline(0.035, linestyle='--', color='gray',
                   alpha=0.5, label='Random baseline')

    plt.suptitle(
        f'MPNN vs RF Active Learning: acquisition function comparison\n'
        f'(batch={BATCH_SIZE}, init={int(100*INIT_FRACTION)}%, {N_SEEDS} seeds)',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.show()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data once
    (X_pool, y_pool, X_test, y_test,
     graphs_pool, graphs_test) = load_all_data()

    # Define conditions to run
    # Start with RF conditions (fast) then MPNN (slow)
    conditions = [
        ('RF',   'entropy'),
        ('RF',   'weighted'),
        ('MPNN', 'entropy'),
        ('MPNN', 'weighted'),
        ('MPNN', 'bald'),
    ]

    all_summary = {}

    for learner, acq in conditions:
        cond_key = f"{learner}_{acq}"
        seed_results = run_one_condition(
            learner, acq,
            X_pool, y_pool, X_test, y_test,
            graphs_pool, graphs_test,
        )
        all_summary[cond_key] = aggregate_seeds(seed_results)

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'mpnn_al_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Plot
    plot_path = os.path.join(RESULTS_DIR, 'mpnn_al_comparison.png')
    plot_results(all_summary, save_path=plot_path)


if __name__ == '__main__':
    main()