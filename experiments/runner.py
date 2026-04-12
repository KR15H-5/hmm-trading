# experiments/runner.py
# run with: python3 experiments/runner.py

import sys
import os
import csv
import json
import time
import random
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.coordinator import Coordinator


# ─────────────────────────────────────────────────────────────────────────────
# Experiment conditions
# ─────────────────────────────────────────────────────────────────────────────

# Primary sweep: 4 switching speeds × 2 meta-learning conditions = 8 conditions
# Each condition runs N_RUNS independent trials
N_RUNS = 50

SWITCHING_SPEEDS = {
    'slow':      {'mean_duration': 20, 'std_duration': 5},
    'medium':    {'mean_duration': 10, 'std_duration': 3},
    'fast':      {'mean_duration':  5, 'std_duration': 2},
    'very_fast': {'mean_duration':  2, 'std_duration': 1},
}

CONDITIONS = []
for speed_name, speed_params in SWITCHING_SPEEDS.items():
    for meta_on in [True, False]:
        CONDITIONS.append({
            'speed_name':    speed_name,
            'mean_duration': speed_params['mean_duration'],
            'std_duration':  speed_params['std_duration'],
            'enable_meta':   meta_on,
            'label': f"{speed_name}_{'meta' if meta_on else 'nometa'}",
        })


# ─────────────────────────────────────────────────────────────────────────────
# Single trial runner — must be a top-level function for multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

def run_single_trial(args):
    """
    Run one trial for one condition.

    Parameters
    ----------
    args : tuple of (condition_dict, trial_seed)

    Returns
    -------
    dict of metrics for this trial
    """
    condition, seed = args

    coord = Coordinator(
        n_sessions     = 100,
        mean_duration  = condition['mean_duration'],
        std_duration   = condition['std_duration'],
        session_length = 60.0,
        n_buyers       = 10,
        seed           = seed,
        enable_meta    = condition['enable_meta'],
        enable_risk    = True,
        hmm_warmup     = 21,
    )

    coord.run()
    summary = coord.summary()

    # detection lag: average sessions between a regime switch and
    # the HMM correctly identifying the new regime
    detection_lag = _compute_detection_lag(coord.results)

    return {
        'condition':    condition['label'],
        'speed':        condition['speed_name'],
        'meta':         condition['enable_meta'],
        'seed':         seed,
        'hmm_accuracy': summary.get('hmm_accuracy', 0.0),
        'total_pnl':    summary.get('total_pnl',    0.0),
        'sharpe':       summary.get('sharpe',        0.0),
        'n_vetoes':     summary.get('n_vetoes',      0),
        'n_retrains':   summary.get('n_retrains',    0),
        'detection_lag': detection_lag,
    }


def _compute_detection_lag(results):
    """
    Compute average detection lag in sessions.

    Detection lag = number of sessions after a regime switch before
    the HMM correctly predicts the new regime.

    A switch is detected when true_regime changes AND the HMM's
    predicted_regime matches the new true_regime.
    """
    live = [r for r in results if r['veto_reason'] != 'warmup']
    if len(live) < 2:
        return 0.0

    lags        = []
    in_lag      = False
    lag_count   = 0
    prev_regime = live[0]['true_regime']

    for r in live[1:]:
        # detect a regime switch
        if r['true_regime'] != prev_regime:
            in_lag    = True
            lag_count = 0

        if in_lag:
            lag_count += 1
            # HMM has caught up when prediction matches new regime
            if r['predicted_regime'] == r['true_regime']:
                lags.append(lag_count)
                in_lag = False

        prev_regime = r['true_regime']

    return round(sum(lags) / len(lags), 2) if lags else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def mean(values):
    return sum(values) / len(values) if values else 0.0

def std(values):
    if len(values) < 2:
        return 0.0
    m   = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)

def confidence_interval_95(values):
    """95% CI using t-distribution approximation (n >= 30 → z ≈ 1.96)."""
    if len(values) < 2:
        return 0.0
    return 1.96 * std(values) / math.sqrt(len(values))

def t_test_two_sample(a, b):
    """
    Welch's t-test p-value approximation.
    Tests whether means of a and b are significantly different.
    Returns p-value (< 0.05 = significant at 95% confidence).
    """
    import math
    na, nb   = len(a), len(b)
    if na < 2 or nb < 2:
        return 1.0
    ma, mb   = mean(a), mean(b)
    sa, sb   = std(a),  std(b)
    se       = math.sqrt(sa**2/na + sb**2/nb)
    if se == 0:
        return 1.0
    t_stat   = abs(ma - mb) / se
    # approximate p-value using normal distribution for large n
    # for n >= 30 per group, this is close enough
    # p = 2 * (1 - Phi(|t|)) where Phi is standard normal CDF
    # approximation: p ≈ exp(-0.717 * t - 0.416 * t^2)
    p_approx = math.exp(-0.717 * t_stat - 0.416 * t_stat**2)
    return round(min(1.0, p_approx), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_experiments(n_runs=N_RUNS, n_workers=4, output_dir='results'):
    """
    Run the full experiment sweep.

    Parameters
    ----------
    n_runs     : trials per condition
    n_workers  : parallel processes
    output_dir : where to save CSV results

    Saves
    -----
    results/raw_results.csv      — one row per trial
    results/summary_results.csv  — one row per condition with stats
    results/pairwise_tests.csv   — t-tests between conditions
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Running {len(CONDITIONS)} conditions × {n_runs} trials "
          f"= {len(CONDITIONS) * n_runs} total trials")
    print(f"Using {n_workers} parallel workers")
    print()

    # build all (condition, seed) pairs
    all_args = []
    for condition in CONDITIONS:
        for trial_idx in range(n_runs):
            seed = hash(condition['label'] + str(trial_idx)) % (2**31)
            all_args.append((condition, seed))

    random.shuffle(all_args)  # shuffle so workers get mixed conditions

    # run all trials
    all_results = []
    start_time  = time.time()
    completed   = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_trial, args): args
                   for args in all_args}

        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                if completed % 10 == 0:
                    elapsed  = time.time() - start_time
                    rate     = completed / elapsed
                    remaining = (len(all_args) - completed) / rate
                    print(f"  {completed}/{len(all_args)} trials complete "
                          f"({remaining:.0f}s remaining)")
            except Exception as e:
                print(f"  Trial failed: {e}")

    elapsed = time.time() - start_time
    print(f"\nAll trials complete in {elapsed:.1f}s")

    # ── save raw results ───────────────────────────────────────────────
    raw_path = os.path.join(output_dir, 'raw_results.csv')
    fieldnames = ['condition', 'speed', 'meta', 'seed',
                  'hmm_accuracy', 'total_pnl', 'sharpe',
                  'n_vetoes', 'n_retrains', 'detection_lag']

    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Raw results saved to {raw_path}")

    # ── compute and save summary statistics ───────────────────────────
    summary_rows = []
    grouped = {}
    for r in all_results:
        key = r['condition']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    for condition_label, trials in grouped.items():
        accuracies  = [t['hmm_accuracy']  for t in trials]
        pnls        = [t['total_pnl']     for t in trials]
        sharpes     = [t['sharpe']        for t in trials]
        lags        = [t['detection_lag'] for t in trials]
        vetoes      = [t['n_vetoes']      for t in trials]
        retrains    = [t['n_retrains']    for t in trials]

        summary_rows.append({
            'condition':       condition_label,
            'speed':           trials[0]['speed'],
            'meta':            trials[0]['meta'],
            'n_trials':        len(trials),
            'accuracy_mean':   round(mean(accuracies),  4),
            'accuracy_std':    round(std(accuracies),   4),
            'accuracy_ci95':   round(confidence_interval_95(accuracies), 4),
            'pnl_mean':        round(mean(pnls),        2),
            'pnl_std':         round(std(pnls),         2),
            'pnl_ci95':        round(confidence_interval_95(pnls), 2),
            'sharpe_mean':     round(mean(sharpes),     4),
            'sharpe_std':      round(std(sharpes),      4),
            'detection_lag_mean': round(mean(lags),     2),
            'detection_lag_std':  round(std(lags),      2),
            'vetoes_mean':     round(mean(vetoes),      2),
            'retrains_mean':   round(mean(retrains),    2),
        })

    summary_path = os.path.join(output_dir, 'summary_results.csv')
    summary_fields = [
        'condition', 'speed', 'meta', 'n_trials',
        'accuracy_mean', 'accuracy_std', 'accuracy_ci95',
        'pnl_mean', 'pnl_std', 'pnl_ci95',
        'sharpe_mean', 'sharpe_std',
        'detection_lag_mean', 'detection_lag_std',
        'vetoes_mean', 'retrains_mean',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary results saved to {summary_path}")

    # ── pairwise t-tests: meta vs no-meta per speed ───────────────────
    test_rows = []
    for speed in SWITCHING_SPEEDS.keys():
        meta_key   = f"{speed}_meta"
        nometa_key = f"{speed}_nometa"
        if meta_key not in grouped or nometa_key not in grouped:
            continue

        meta_acc   = [t['hmm_accuracy'] for t in grouped[meta_key]]
        nometa_acc = [t['hmm_accuracy'] for t in grouped[nometa_key]]
        meta_pnl   = [t['total_pnl']    for t in grouped[meta_key]]
        nometa_pnl = [t['total_pnl']    for t in grouped[nometa_key]]

        test_rows.append({
            'speed':             speed,
            'meta_accuracy':     round(mean(meta_acc),   4),
            'nometa_accuracy':   round(mean(nometa_acc), 4),
            'accuracy_diff':     round(mean(meta_acc) - mean(nometa_acc), 4),
            'accuracy_pvalue':   t_test_two_sample(meta_acc, nometa_acc),
            'meta_pnl':          round(mean(meta_pnl),   2),
            'nometa_pnl':        round(mean(nometa_pnl), 2),
            'pnl_diff':          round(mean(meta_pnl) - mean(nometa_pnl), 2),
            'pnl_pvalue':        t_test_two_sample(meta_pnl, nometa_pnl),
        })

    tests_path = os.path.join(output_dir, 'pairwise_tests.csv')
    test_fields = [
        'speed',
        'meta_accuracy', 'nometa_accuracy', 'accuracy_diff', 'accuracy_pvalue',
        'meta_pnl', 'nometa_pnl', 'pnl_diff', 'pnl_pvalue',
    ]
    with open(tests_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=test_fields)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"Pairwise tests saved to {tests_path}")

    # ── print summary table ───────────────────────────────────────────
    print()
    print("=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Condition':<25} {'Accuracy':<12} {'CI95':<10} "
          f"{'PnL':<10} {'Sharpe':<10} {'DetLag'}")
    print("-" * 75)

    for row in sorted(summary_rows, key=lambda r: (r['speed'], r['meta'])):
        meta_str = 'meta' if row['meta'] else 'nometa'
        label    = f"{row['speed']}_{meta_str}"
        print(f"{label:<25} "
              f"{row['accuracy_mean']:.1%}{'':>4} "
              f"±{row['accuracy_ci95']:.1%}{'':>2} "
              f"{row['pnl_mean']:<10.1f} "
              f"{row['sharpe_mean']:<10.4f} "
              f"{row['detection_lag_mean']:.2f}")

    print()
    print("Meta-learning effect (accuracy improvement WITH vs WITHOUT):")
    for row in test_rows:
        sig = '**' if row['accuracy_pvalue'] < 0.05 else '  '
        print(f"  {row['speed']:<12} "
              f"diff={row['accuracy_diff']:+.1%} "
              f"p={row['accuracy_pvalue']:.4f} {sig}")

    return all_results, summary_rows, test_rows


if __name__ == '__main__':
    run_experiments(n_runs=N_RUNS, n_workers=4, output_dir='results')