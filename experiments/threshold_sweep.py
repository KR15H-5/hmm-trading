import sys
import os
import csv
import time
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system.coordinator import Coordinator

THRESHOLDS = [0.20, 0.35, 0.50, 0.60]
N_RUNS     = 50

def run_single(args):
    thr, seed = args
    coord = Coordinator(
        n_sessions        = 100,
        mean_duration     = 10,
        std_duration      = 3,
        session_length    = 300.0,
        n_buyers          = 10,
        seed              = seed,
        enable_meta       = True,
        enable_risk       = True,
        hmm_warmup        = 21,
        retrain_threshold = thr,
    )
    coord.run()
    summary = coord.summary()

    live = [r for r in coord.results if r['veto_reason'] != 'warmup']
    lags, in_lag, lag_count = [], False, 0
    if live:
        prev = live[0]['true_regime']
        for r in live[1:]:
            if r['true_regime'] != prev:
                in_lag, lag_count = True, 0
            if in_lag:
                lag_count += 1
                if r['predicted_regime'] == r['true_regime']:
                    lags.append(lag_count); in_lag = False
            prev = r['true_regime']
    detection_lag = round(sum(lags)/len(lags), 2) if lags else 0.0

    return {
        'threshold':          thr,
        'seed':               seed,
        'hmm_accuracy':       summary.get('hmm_accuracy', 0.0),
        'total_pnl':          summary.get('total_pnl', 0.0),
        'sharpe':             summary.get('sharpe', 0.0),
        'n_retrains':         summary.get('n_retrains', 0),
        'detection_lag':      detection_lag,
        'acc_trending':       summary.get('acc_trending', 0.0),
        'acc_mean_reverting': summary.get('acc_mean_reverting', 0.0),
        'acc_volatile':       summary.get('acc_volatile', 0.0),
    }


def mean(v): return sum(v)/len(v) if v else 0.0
def std_(v):
    if len(v) < 2: return 0.0
    m = mean(v); return math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
def ci95(v):  return 1.96*std_(v)/math.sqrt(len(v)) if len(v) > 1 else 0.0


def main(n_workers=4, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    args_list = []
    for thr in THRESHOLDS:
        for k in range(N_RUNS):
            seed = hash(f'thr_{thr}_{k}') % (2**31)
            args_list.append((thr, seed))
    random.shuffle(args_list)

    total = len(args_list)
    print(f"Threshold sweep: {len(THRESHOLDS)} thresholds x {N_RUNS} runs = {total}")
    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_single, a): a for a in args_list}
        done = 0
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                print('error:', e)
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - start
                rate    = done / elapsed if elapsed > 0 else 0
                remain  = (total - done) / rate if rate > 0 else 0
                print(f'  {done}/{total}  elapsed {elapsed:.0f}s  '
                      f'remaining ~{remain:.0f}s  rate {rate:.2f}/s')

    raw_path = os.path.join(output_dir, 'threshold_sweep_raw.csv')
    fields = ['threshold','seed','hmm_accuracy','total_pnl','sharpe',
              'n_retrains','detection_lag',
              'acc_trending','acc_mean_reverting','acc_volatile']
    with open(raw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(results)

    summary_rows = []
    for thr in THRESHOLDS:
        sub = [r for r in results if r['threshold'] == thr]
        row = {'threshold': thr, 'n_trials': len(sub)}
        for field in ['hmm_accuracy','total_pnl','sharpe','n_retrains',
                      'detection_lag','acc_trending','acc_mean_reverting',
                      'acc_volatile']:
            vals = [float(r[field]) for r in sub]
            row[f'{field}_mean'] = round(mean(vals), 4)
            row[f'{field}_std']  = round(std_(vals), 4)
            row[f'{field}_ci95'] = round(ci95(vals), 4)
        summary_rows.append(row)

    sum_path = os.path.join(output_dir, 'threshold_sweep_summary.csv')
    sum_fields = ['threshold','n_trials']
    for f in ['hmm_accuracy','total_pnl','sharpe','n_retrains',
              'detection_lag','acc_trending','acc_mean_reverting','acc_volatile']:
        sum_fields += [f'{f}_mean', f'{f}_std', f'{f}_ci95']
    with open(sum_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        w.writeheader(); w.writerows(summary_rows)

    print()
    print(f"  {'thr':>5} {'acc':>7} {'lag':>7} {'pnl':>7} {'retrains':>10}")
    print(f"  {'-'*40}")
    for row in summary_rows:
        print(f"  {row['threshold']:>5} "
              f"{row['hmm_accuracy_mean']:>6.1%} "
              f"{row['detection_lag_mean']:>7.3f} "
              f"{row['total_pnl_mean']:>7.0f} "
              f"{row['n_retrains_mean']:>10.2f}")


if __name__ == '__main__':
    main(n_workers=4, output_dir='results')
