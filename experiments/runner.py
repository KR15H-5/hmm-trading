# experiments/runner.py

import sys
import os
import csv
import time
import random
import math
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.coordinator import Coordinator


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


def run_single_trial(args):
    condition, seed = args
    coord = Coordinator(
        n_sessions     = 100,
        mean_duration  = condition['mean_duration'],
        std_duration   = condition['std_duration'],
        session_length = 300.0,
        n_buyers       = 10,
        seed           = seed,
        enable_meta    = condition['enable_meta'],
        enable_risk    = True,
        hmm_warmup     = 21,
    )
    coord.run()
    summary = coord.summary()
    detection_lag = _compute_detection_lag(coord.results)
    return {
        'condition':     condition['label'],
        'speed':         condition['speed_name'],
        'meta':          condition['enable_meta'],
        'seed':          seed,
        'hmm_accuracy':  summary.get('hmm_accuracy',  0.0),
        'total_pnl':     summary.get('total_pnl',     0.0),
        'sharpe':        summary.get('sharpe',         0.0),
        'n_vetoes':      summary.get('n_vetoes',       0),
        'n_retrains':    summary.get('n_retrains',     0),
        'detection_lag': detection_lag,
    }


def _compute_detection_lag(results):
    live = [r for r in results if r['veto_reason'] != 'warmup']
    if len(live) < 2:
        return 0.0
    lags, in_lag, lag_count = [], False, 0
    prev_regime = live[0]['true_regime']
    for r in live[1:]:
        if r['true_regime'] != prev_regime:
            in_lag, lag_count = True, 0
        if in_lag:
            lag_count += 1
            if r['predicted_regime'] == r['true_regime']:
                lags.append(lag_count)
                in_lag = False
        prev_regime = r['true_regime']
    return round(sum(lags) / len(lags), 2) if lags else 0.0


def mean(v):   return sum(v) / len(v) if v else 0.0
def std(v):
    if len(v) < 2: return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
def ci95(v):   return 1.96 * std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
def t_test(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 1.0
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    se = math.sqrt(sa**2/na + sb**2/nb)
    if se == 0: return 1.0
    t = abs(ma - mb) / se
    return round(min(1.0, math.exp(-0.717*t - 0.416*t*t)), 4)


# ── terminal UI helpers ────────────────────────────────────────────────────

BAR_WIDTH = 30
COLS = 80

def clear_lines(n):
    for _ in range(n):
        sys.stdout.write('\033[F\033[K')

def bar(filled, total, width=BAR_WIDTH):
    if total == 0: return '[' + ' ' * width + ']'
    n = int(width * filled / total)
    return '[' + '█' * n + '░' * (width - n) + ']'

def render(conditions_progress, total_done, total_all, start_time, results_so_far):
    elapsed   = time.time() - start_time
    rate      = total_done / elapsed if elapsed > 0 else 0.001
    remaining = (total_all - total_done) / rate if rate > 0 else 0

    lines = []
    lines.append(f"  HMM Trading Experiment")
    lines.append(f"  {'─' * (COLS - 4)}")

    for cond in CONDITIONS:
        label    = cond['label']
        done     = conditions_progress.get(label, 0)
        pct      = done / N_RUNS
        b        = bar(done, N_RUNS)

        # live accuracy if we have results
        cond_results = [r for r in results_so_far if r['condition'] == label]
        if cond_results:
            acc = mean([r['hmm_accuracy'] for r in cond_results])
            acc_str = f"  acc={acc:.1%}"
        else:
            acc_str = ""

        speed_tag = label.replace('_meta','').replace('_nometa','')
        meta_tag  = 'meta  ' if 'nometa' not in label else 'nometa'
        lines.append(f"  {speed_tag:<10} {meta_tag}  {b} {done:>3}/{N_RUNS}{acc_str}")

    lines.append(f"  {'─' * (COLS - 4)}")

    overall_bar = bar(total_done, total_all, width=BAR_WIDTH + 10)
    lines.append(f"  overall  {overall_bar}  {total_done}/{total_all}")

    mins_rem = int(remaining // 60)
    secs_rem = int(remaining % 60)
    mins_ela = int(elapsed // 60)
    secs_ela = int(elapsed % 60)
    lines.append(f"  elapsed {mins_ela:02d}:{secs_ela:02d}   remaining ~{mins_rem:02d}:{secs_rem:02d}   "
                 f"rate {rate:.1f} trials/s")

    return lines


def run_experiments(n_runs=N_RUNS, n_workers=4, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    total_all = len(CONDITIONS) * n_runs
    all_args  = []
    for condition in CONDITIONS:
        for trial_idx in range(n_runs):
            seed = hash(condition['label'] + str(trial_idx)) % (2**31)
            all_args.append((condition, seed))
    random.shuffle(all_args)

    conditions_progress = {c['label']: 0 for c in CONDITIONS}
    all_results         = []
    start_time          = time.time()
    total_done          = 0

    # print initial render
    initial_lines = render(conditions_progress, 0, total_all, start_time, [])
    print('\n'.join(initial_lines))
    n_lines = len(initial_lines)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_trial, args): args
                   for args in all_args}

        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                conditions_progress[result['condition']] += 1
                total_done += 1
            except Exception as e:
                total_done += 1

            # redraw
            clear_lines(n_lines)
            new_lines = render(conditions_progress, total_done, total_all,
                               start_time, all_results)
            print('\n'.join(new_lines))
            n_lines = len(new_lines)

    # ── save raw ───────────────────────────────────────────────────────────
    raw_path   = os.path.join(output_dir, 'raw_results.csv')
    fieldnames = ['condition','speed','meta','seed','hmm_accuracy',
                  'total_pnl','sharpe','n_vetoes','n_retrains','detection_lag']
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # ── summary ────────────────────────────────────────────────────────────
    grouped = {}
    for r in all_results:
        grouped.setdefault(r['condition'], []).append(r)

    summary_rows = []
    for label, trials in grouped.items():
        accs  = [t['hmm_accuracy']  for t in trials]
        pnls  = [t['total_pnl']     for t in trials]
        shps  = [t['sharpe']        for t in trials]
        lags  = [t['detection_lag'] for t in trials]
        vets  = [t['n_vetoes']      for t in trials]
        rets  = [t['n_retrains']    for t in trials]
        summary_rows.append({
            'condition':          label,
            'speed':              trials[0]['speed'],
            'meta':               trials[0]['meta'],
            'n_trials':           len(trials),
            'accuracy_mean':      round(mean(accs), 4),
            'accuracy_std':       round(std(accs),  4),
            'accuracy_ci95':      round(ci95(accs), 4),
            'pnl_mean':           round(mean(pnls), 2),
            'pnl_std':            round(std(pnls),  2),
            'pnl_ci95':           round(ci95(pnls), 2),
            'sharpe_mean':        round(mean(shps), 4),
            'sharpe_std':         round(std(shps),  4),
            'detection_lag_mean': round(mean(lags), 2),
            'detection_lag_std':  round(std(lags),  2),
            'vetoes_mean':        round(mean(vets),  2),
            'retrains_mean':      round(mean(rets),  2),
        })

    summary_path = os.path.join(output_dir, 'summary_results.csv')
    summary_fields = [
        'condition','speed','meta','n_trials',
        'accuracy_mean','accuracy_std','accuracy_ci95',
        'pnl_mean','pnl_std','pnl_ci95',
        'sharpe_mean','sharpe_std',
        'detection_lag_mean','detection_lag_std',
        'vetoes_mean','retrains_mean',
    ]
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # ── pairwise ───────────────────────────────────────────────────────────
    test_rows = []
    for speed in SWITCHING_SPEEDS:
        mk, nk = f'{speed}_meta', f'{speed}_nometa'
        if mk not in grouped or nk not in grouped: continue
        ma = [t['hmm_accuracy'] for t in grouped[mk]]
        na = [t['hmm_accuracy'] for t in grouped[nk]]
        mp = [t['total_pnl']    for t in grouped[mk]]
        np_ = [t['total_pnl']   for t in grouped[nk]]
        test_rows.append({
            'speed':           speed,
            'meta_accuracy':   round(mean(ma),  4),
            'nometa_accuracy': round(mean(na),  4),
            'accuracy_diff':   round(mean(ma) - mean(na), 4),
            'accuracy_pvalue': t_test(ma, na),
            'meta_pnl':        round(mean(mp),  2),
            'nometa_pnl':      round(mean(np_), 2),
            'pnl_diff':        round(mean(mp) - mean(np_), 2),
            'pnl_pvalue':      t_test(mp, np_),
        })

    tests_path = os.path.join(output_dir, 'pairwise_tests.csv')
    with open(tests_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'speed','meta_accuracy','nometa_accuracy','accuracy_diff','accuracy_pvalue',
            'meta_pnl','nometa_pnl','pnl_diff','pnl_pvalue'])
        writer.writeheader()
        writer.writerows(test_rows)

    # ── final summary ──────────────────────────────────────────────────────
    print(f"\n  results saved to {output_dir}/")
    print(f"\n  {'condition':<25} {'accuracy':>10}  {'±ci95':>8}  {'pnl':>8}  {'lag':>6}")
    print(f"  {'─'*62}")
    for row in sorted(summary_rows, key=lambda r: (r['speed'], str(r['meta']))):
        print(f"  {row['condition']:<25} "
              f"{row['accuracy_mean']:>9.1%}  "
              f"±{row['accuracy_ci95']:>6.1%}  "
              f"{row['pnl_mean']:>8.0f}  "
              f"{row['detection_lag_mean']:>6.2f}")

    print(f"\n  meta-learning effect:")
    for row in test_rows:
        sig = '**' if row['accuracy_pvalue'] < 0.05 else '  '
        print(f"  {row['speed']:<12} diff={row['accuracy_diff']:+.1%}  p={row['accuracy_pvalue']:.4f} {sig}")

    return all_results, summary_rows, test_rows


if __name__ == '__main__':
    run_experiments(n_runs=N_RUNS, n_workers=4, output_dir='results')