# test_meta_learner.py
# run with: python3 test_meta_learner.py

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import BSE
from models.hmm_detector import HMMDetector, extract_features
from models.meta_learner  import MetaLearner

REGIME_SCHEDULES = {
    'trending':       {'supply': (82, 100), 'demand': (95, 120),
                       'stepmode': 'jittered', 'drift': 0.4},
    'mean_reverting': {'supply': (80, 100), 'demand': (100, 120),
                       'stepmode': 'jittered', 'drift': 0.0},
    'volatile':       {'supply': (60, 100), 'demand': (100, 140),
                       'stepmode': 'random',   'drift': 0.0},
}

def build_schedule(regime, start_time, end_time, drift_offset=0.0):
    sched = REGIME_SCHEDULES[regime]
    s_lo  = max(1,   int(sched['supply'][0] + drift_offset))
    s_hi  = max(1,   int(sched['supply'][1] + drift_offset))
    d_lo  = max(1,   int(sched['demand'][0] + drift_offset))
    d_hi  = min(500, int(sched['demand'][1] + drift_offset))
    return ([{'from': start_time, 'to': end_time,
               'ranges': [(s_lo, s_hi)], 'stepmode': sched['stepmode']}],
            [{'from': start_time, 'to': end_time,
               'ranges': [(d_lo, d_hi)], 'stepmode': sched['stepmode']}])

def run_session(regime, session_idx, drift_offset=0.0):
    sup, dem = build_schedule(regime, 0.0, 60.0, drift_offset)
    order_schedule = {'sup': sup, 'dem': dem,
                      'interval': 5.0, 'timemode': 'drip-fixed'}
    trader_spec    = {'buyers':  [('ZIC', 5), ('ZIP', 5)],
                      'sellers': [('ZIC', 5), ('ZIP', 5)],
                      'proptraders': []}
    dump_flags     = {'dump_blotters': False, 'dump_lobs': False,
                      'dump_strats':   False, 'dump_avgbals': False,
                      'dump_tape':     True}
    random.seed(session_idx * 1000)
    BSE.market_session(f'sess_{session_idx:04d}', 0.0, 60.0,
                       trader_spec, order_schedule, dump_flags, False)
    tape_file = f'sess_{session_idx:04d}_tape.csv'
    trades = []
    with open(tape_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0].strip() == 'TRD':
                trades.append({'type': 'Trade',
                                'time':  float(parts[1].strip()),
                                'price': int(parts[2].strip())})
    os.remove(tape_file)
    return trades


if __name__ == '__main__':

    # ── Phase 1: train HMM on 60 warmup sessions ─────────────────────
    print("=" * 60)
    print("Phase 1: warmup training — 60 sessions")
    print("=" * 60)

    detector = HMMDetector(n_states=3, n_iter=100, warmup=20)
    training_order = []
    for _ in range(20):
        training_order += ['trending', 'mean_reverting', 'volatile']

    for i, regime in enumerate(training_order):
        trades   = run_session(regime, session_idx=i)
        features = extract_features(trades)
        detector.add_observation(features)

    detector.train()
    print(f"HMM trained. State mapping: {detector.state_to_regime}")

    # ── Phase 2: run 40 sessions with meta-learner active ────────────
    print()
    print("=" * 60)
    print("Phase 2: 40 live sessions with meta-learner")
    print("Simulating fast switching — regimes change every 2-3 sessions")
    print("=" * 60)

    meta = MetaLearner(
        detector          = detector,
        error_window      = 10,
        retrain_threshold = 0.35,
        cooldown          = 5,
        retrain_window    = 30,
        enabled           = True,
    )

    # fast switching sequence — changes every 2-3 sessions
    fast_sequence = (
        ['trending']       * 3 +
        ['volatile']       * 2 +
        ['mean_reverting'] * 3 +
        ['trending']       * 2 +
        ['volatile']       * 3 +
        ['mean_reverting'] * 2 +
        ['trending']       * 3 +
        ['volatile']       * 2 +
        ['mean_reverting'] * 3 +
        ['trending']       * 2 +
        ['volatile']       * 2 +
        ['mean_reverting'] * 2 +
        ['trending']       * 2 +
        ['volatile']       * 2 +
        ['mean_reverting'] * 2 +
        ['volatile']       * 2
    )

    print(f"\n{'Sess':<6} {'True':<18} {'Predicted':<18} "
          f"{'Conf':<8} {'ErrRate':<10} {'Retrain':<8} {'OK'}")
    print("-" * 75)

    correct = 0
    total   = 0

    for i, true_regime in enumerate(fast_sequence):
        trades   = run_session(true_regime, session_idx=100 + i)
        features = extract_features(trades)
        result   = detector.predict(features)
        outcome  = meta.record(result['regime'], true_regime, features)

        if outcome['correct']:
            correct += 1
        total += 1

        print(f"{i:<6} {true_regime:<18} {result['regime']:<18} "
              f"{result['confidence']:.2f}{'':>3} "
              f"{outcome['error_rate']:.2f}{'':>5} "
              f"{'YES' if outcome['retrained'] else '':<8} "
              f"{'✓' if outcome['correct'] else '✗'}")

    print()
    print(f"Overall accuracy  : {correct}/{total} = {correct/total:.1%}")
    print(f"Total retrains    : {meta.n_retrains()}")
    print(f"Retrain log       : {meta.retrain_log}")

    # ── Phase 3: compare WITH vs WITHOUT meta-learner ─────────────────
    print()
    print("=" * 60)
    print("Phase 3: same sessions WITHOUT meta-learner (ablation)")
    print("=" * 60)

    # retrain a fresh detector — same warmup
    detector2 = HMMDetector(n_states=3, n_iter=100, warmup=20)
    for i, regime in enumerate(training_order):
        trades   = run_session(regime, session_idx=i)
        features = extract_features(trades)
        detector2.add_observation(features)
    detector2.train()

    meta_off = MetaLearner(
        detector  = detector2,
        enabled   = False,   # ← meta-learner disabled
    )

    correct2 = 0
    for i, true_regime in enumerate(fast_sequence):
        trades   = run_session(true_regime, session_idx=100 + i)
        features = extract_features(trades)
        result   = detector2.predict(features)
        outcome  = meta_off.record(result['regime'], true_regime, features)
        if outcome['correct']:
            correct2 += 1

    print(f"Accuracy WITHOUT meta-learner: {correct2}/{total} = {correct2/total:.1%}")
    print(f"Accuracy WITH    meta-learner: {correct}/{total} = {correct/total:.1%}")
    print(f"Improvement: {(correct - correct2)/total:+.1%}")