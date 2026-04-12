# test_hmm.py

import sys
import os
import random
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import BSE
from models.hmm_detector import HMMDetector, extract_features

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

    detector = HMMDetector(n_states=3, n_iter=100, warmup=20)

    print("=" * 60)
    print("Phase 1: training — 20 sessions per regime (60 total)")
    print("=" * 60)

    # interleave regimes so HMM sees realistic switching patterns
    training_order = []
    for _ in range(20):
        training_order += ['trending', 'mean_reverting', 'volatile']

    for i, regime in enumerate(training_order):
        trades   = run_session(regime, session_idx=i)
        features = extract_features(trades)
        detector.add_observation(features)

    print("Training HMM on 60 observations...")
    success = detector.train()
    print(f"Training succeeded : {success}")
    print(f"State mapping      : {detector.state_to_regime}")

    # show learned emission means so we can see if states are separating
    if detector.model is not None:
        print(f"\nLearned emission means (momentum, volatility, range):")
        for state, regime in detector.state_to_regime.items():
            m = detector.model.means_[state]
            print(f"  {regime:18s} → [{m[0]:+.4f},  {m[1]:.4f},  {m[2]:.4f}]")

    print()
    print("=" * 60)
    print("Phase 2: predict 15 new sessions (5 per regime)")
    print("=" * 60)
    print(f"{'Session':<10} {'True':<18} {'Predicted':<18} "
          f"{'Confidence':<12} {'Correct'}")
    print("-" * 65)

    correct   = 0
    total     = 0
    test_order = ['trending'] * 5 + ['mean_reverting'] * 5 + ['volatile'] * 5

    for i, true_regime in enumerate(test_order):
        trades   = run_session(true_regime, session_idx=200 + i)
        features = extract_features(trades)
        result   = detector.predict(features)

        is_correct = result['regime'] == true_regime
        if is_correct:
            correct += 1
        total += 1

        print(f"{i:<10} {true_regime:<18} {result['regime']:<18} "
              f"{result['confidence']:.3f}{'':>6} "
              f"{'✓' if is_correct else '✗'}")

    print()
    print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")
    print()
    print("=" * 60)
    print("Phase 3: online update with rolling window")
    print("=" * 60)

    for i in range(5):
        trades   = run_session('volatile', session_idx=300 + i)
        features = extract_features(trades)
        updated  = detector.update(features, window=30)
        print(f"  Update {i}: retrained={updated} "
              f"history={len(detector.observation_history)}")

    print()
    print("All HMM tests complete.")