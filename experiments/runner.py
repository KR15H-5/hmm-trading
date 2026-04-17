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

SESSION_LENGTHS = [60.0, 120.0, 180.0, 300.0]

CONDITIONS = []

for speed_name, speed_params in SWITCHING_SPEEDS.items():
    for meta_on in [True, False]:
        CONDITIONS.append({
            'speed_name':     speed_name,
            'mean_duration':  speed_params['mean_duration'],
            'std_duration':   speed_params['std_duration'],
            'enable_meta':    meta_on,
            'session_length': 300.0,
            'experiment':     'speed_sweep',
            'label': f"{speed_name}_{'meta' if meta_on else 'nometa'}",
        })

for sl in SESSION_LENGTHS:
    CONDITIONS.append({
        'speed_name':     'medium',
        'mean_duration':  10,
        'std_duration':   3,
        'enable_meta':    False,
        'session_length': sl,
        'experiment':     'session_sweep',
        'label':          f"sesslen_{int(sl)}",
    })


def run_single_trial(args):
    condition, seed = args
    coord = Coordinator(
        n_sessions     = 100,
        mean_duration  = condition['mean_duration'],
        std_duration   = condition['std_duration'],
        session_length = condition['session_length'],
        n_buyers       = 10,
        seed           = seed,
        enable_meta    = condition['enable_meta'],
        enable_risk    = True,
        hmm_warmup     = 21,
    )
    coord.run()
    summary = coord.summary()
    detection_lag = _compute_detection_lag(coord.results)
    per_trans_lag = summary.get('per_transition_lag', {})

    return {
        'condition':          condition['label'],
        'experiment':         condition['experiment'],
        'speed':              condition['speed_name'],
        'session_length':     condition['session_length'],
        'meta':               condition['enable_meta'],
        'seed':               seed,
        'hmm_accuracy':       summary.get('hmm_accuracy',      0.0),
        'total_pnl':          summary.get('total_pnl',         0.0),
        'sharpe':             summary.get('sharpe',            0.0),
        'n_vetoes':           summary.get('n_vetoes',          0),
        'n_retrains':         summary.get('n_retrains',        0),
        'detection_lag':      detection_lag,
        'acc_trending':       summary.get('acc_trending',      0.0),
        'acc_mean_reverting': summary.get('acc_mean_reverting',0.0),
        'acc_volatile':       summary.get('acc_volatile',      0.0),
        'lag_trending_to_mean_rev':   per_trans_lag.get('trending_to_mean_reverting', 0.0),
        'lag_trending_to_volatile':   per_trans_lag.get('trending_to_volatile', 0.0),
        'lag_mean_rev_to_trending':   per_trans_lag.get('mean_reverting_to_trending', 0.0),
        'lag_mean_rev_to_volatile':   per_trans_lag.get('mean_reverting_to_volatile', 0.0),
        'lag_volatile_to_trending':   per_trans_lag.get('volatile_to_trending', 0.0),
        'lag_volatile_to_mean_rev':   per_trans_lag.get('volatile_to_mean_reverting', 0.0),
        'pnl_when_correct':   summary.get('pnl_when_correct',   0.0),
        'pnl_when_incorrect': summary.get('pnl_when_incorrect', 0.0),
        'avg_pnl_correct':    summary.get('avg_pnl_correct',    0.0),
        'avg_pnl_incorrect':  summary.get('avg_pnl_incorrect',  0.0),
        'conf_at_transition': summary.get('conf_at_transition', 0.0),
        'conf_at_stable':     summary.get('conf_at_stable',     0.0),
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
    return round(sum(lags)/len(lags), 2) if lags else 0.0


def mean(v):   return sum(v)/len(v) if v else 0.0
def std(v):
    if len(v) < 2: return 0.0
    m = mean(v)
    return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
def ci95(v):   return 1.96*std(v)/math.sqrt(len(v)) if len(v) > 1 else 0.0
def t_test(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return 1.0
    ma, mb = mean(a), mean(b)
    sa, sb = std(a), std(b)
    se = math.sqrt(sa**2/na + sb**2/nb)
    if se == 0: return 1.0
    t = abs(ma-mb)/se
    return round(min(1.0, math.exp(-0.717*t - 0.416*t*t)), 4)


BAR_WIDTH = 26
COLS = 82

def clear_lines(n):
    for _ in range(n):
        sys.stdout.write('\033[F\033[K')

def bar(filled, total, width=BAR_WIDTH):
    if total == 0: return '[' + ' '*width + ']'
    n = int(width * filled / total)
    return '[' + '█'*n + '░'*(width-n) + ']'

def render(conditions_progress, total_done, total_all, start_time, results_so_far):
    elapsed   = time.time() - start_time
    rate      = total_done / elapsed if elapsed > 0 else 0.001
    remaining = (total_all - total_done) / rate if rate > 0 else 0
    lines = []
    lines.append(f"  HMM Trading Experiment")
    lines.append(f"  {'─'*(COLS-4)}")
    for cond in CONDITIONS:
        label = cond['label']
        done  = conditions_progress.get(label, 0)
        b     = bar(done, N_RUNS)
        cr    = [r for r in results_so_far if r['condition'] == label]
        acc_s = f" acc={mean([r['hmm_accuracy'] for r in cr]):.1%}" if cr else ""
        lag_s = f" lag={mean([r['detection_lag'] for r in cr]):.2f}" if cr else ""
        lines.append(f"  {label:<22} {b} {done:>3}/{N_RUNS}{acc_s}{lag_s}")
    lines.append(f"  {'─'*(COLS-4)}")
    ob = bar(total_done, total_all, width=BAR_WIDTH+12)
    lines.append(f"  overall  {ob}  {total_done}/{total_all}")
    mr,sr = int(remaining//60),int(remaining%60)
    me,se = int(elapsed//60),int(elapsed%60)
    lines.append(f"  elapsed {me:02d}:{se:02d}   remaining ~{mr:02d}:{sr:02d}   rate {rate:.1f}/s")
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
    all_results = []
    start_time  = time.time()
    total_done  = 0

    initial_lines = render(conditions_progress, 0, total_all, start_time, [])
    print('\n'.join(initial_lines))
    n_lines = len(initial_lines)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_single_trial, args): args for args in all_args}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                conditions_progress[result['condition']] += 1
                total_done += 1
            except Exception as e:
                print(f"  Trial error: {e}")
                total_done += 1
            clear_lines(n_lines)
            new_lines = render(conditions_progress, total_done, total_all,
                               start_time, all_results)
            print('\n'.join(new_lines))
            n_lines = len(new_lines)

    # save raw
    raw_path = os.path.join(output_dir, 'raw_results.csv')
    fieldnames = [
        'condition','experiment','speed','session_length','meta','seed',
        'hmm_accuracy','total_pnl','sharpe','n_vetoes','n_retrains','detection_lag',
        'acc_trending','acc_mean_reverting','acc_volatile',
        'lag_trending_to_mean_rev','lag_trending_to_volatile',
        'lag_mean_rev_to_trending','lag_mean_rev_to_volatile',
        'lag_volatile_to_trending','lag_volatile_to_mean_rev',
        'pnl_when_correct','pnl_when_incorrect',
        'avg_pnl_correct','avg_pnl_incorrect',
        'conf_at_transition','conf_at_stable',
    ]
    with open(raw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # summary
    grouped = {}
    for r in all_results:
        grouped.setdefault(r['condition'], []).append(r)

    numeric_fields = [
        'hmm_accuracy','total_pnl','sharpe','n_vetoes','n_retrains','detection_lag',
        'acc_trending','acc_mean_reverting','acc_volatile',
        'lag_trending_to_mean_rev','lag_trending_to_volatile',
        'lag_mean_rev_to_trending','lag_mean_rev_to_volatile',
        'lag_volatile_to_trending','lag_volatile_to_mean_rev',
        'pnl_when_correct','pnl_when_incorrect',
        'avg_pnl_correct','avg_pnl_incorrect',
        'conf_at_transition','conf_at_stable',
    ]

    summary_rows = []
    for label, trials in grouped.items():
        row = {
            'condition':      label,
            'experiment':     trials[0]['experiment'],
            'speed':          trials[0]['speed'],
            'session_length': trials[0]['session_length'],
            'meta':           trials[0]['meta'],
            'n_trials':       len(trials),
        }
        for field in numeric_fields:
            vals = [float(t[field]) for t in trials
                    if t.get(field) is not None and t[field] != '']
            if vals:
                row[f'{field}_mean'] = round(mean(vals), 4)
                row[f'{field}_std']  = round(std(vals),  4)
                row[f'{field}_ci95'] = round(ci95(vals), 4)
            else:
                row[f'{field}_mean'] = 0.0
                row[f'{field}_std']  = 0.0
                row[f'{field}_ci95'] = 0.0
        summary_rows.append(row)

    summary_fields = ['condition','experiment','speed','session_length','meta','n_trials']
    for f in numeric_fields:
        summary_fields += [f'{f}_mean', f'{f}_std', f'{f}_ci95']

    summary_path = os.path.join(output_dir, 'summary_results.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # pairwise tests
    test_rows = []
    for speed in SWITCHING_SPEEDS:
        mk, nk = f'{speed}_meta', f'{speed}_nometa'
        if mk not in grouped or nk not in grouped: continue
        ma  = [float(t['hmm_accuracy'])  for t in grouped[mk]]
        na  = [float(t['hmm_accuracy'])  for t in grouped[nk]]
        ml  = [float(t['detection_lag']) for t in grouped[mk]]
        nl  = [float(t['detection_lag']) for t in grouped[nk]]
        mp  = [float(t['total_pnl'])     for t in grouped[mk]]
        np_ = [float(t['total_pnl'])     for t in grouped[nk]]
        test_rows.append({
            'speed':           speed,
            'meta_accuracy':   round(mean(ma),4),
            'nometa_accuracy': round(mean(na),4),
            'accuracy_diff':   round(mean(ma)-mean(na),4),
            'accuracy_pvalue': t_test(ma,na),
            'meta_lag':        round(mean(ml),4),
            'nometa_lag':      round(mean(nl),4),
            'lag_diff':        round(mean(ml)-mean(nl),4),
            'lag_pvalue':      t_test(ml,nl),
            'meta_pnl':        round(mean(mp),2),
            'nometa_pnl':      round(mean(np_),2),
            'pnl_diff':        round(mean(mp)-mean(np_),2),
            'pnl_pvalue':      t_test(mp,np_),
        })

    tests_path = os.path.join(output_dir, 'pairwise_tests.csv')
    with open(tests_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'speed','meta_accuracy','nometa_accuracy','accuracy_diff','accuracy_pvalue',
            'meta_lag','nometa_lag','lag_diff','lag_pvalue',
            'meta_pnl','nometa_pnl','pnl_diff','pnl_pvalue'])
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"\n  Results saved to {output_dir}/")
    print(f"\n  {'condition':<22} {'acc':>7} {'lag':>7} {'pnl':>7} {'acc_T':>7} {'acc_MR':>7} {'acc_V':>7}")
    print(f"  {'─'*62}")
    for row in sorted(summary_rows, key=lambda r: (r['experiment'], r['speed'], str(r['meta']))):
        print(f"  {row['condition']:<22} "
              f"{row['hmm_accuracy_mean']:>6.1%} "
              f"{row['detection_lag_mean']:>7.3f} "
              f"{row['total_pnl_mean']:>7.0f} "
              f"{row['acc_trending_mean']:>7.1%} "
              f"{row['acc_mean_reverting_mean']:>7.1%} "
              f"{row['acc_volatile_mean']:>7.1%}")

    print(f"\n  Meta-learning effect:")
    for row in test_rows:
        sa = '**' if row['accuracy_pvalue']<0.05 else '  '
        sl = '**' if row['lag_pvalue']<0.05 else '  '
        sp = '**' if row['pnl_pvalue']<0.05 else '  '
        print(f"  {row['speed']:<12} acc={row['accuracy_diff']:+.1%}{sa} "
              f"lag={row['lag_diff']:+.3f}{sl} pnl={row['pnl_diff']:+.0f}{sp}")

    return all_results, summary_rows, test_rows


if __name__ == '__main__':
    run_experiments(n_runs=N_RUNS, n_workers=4, output_dir='results')