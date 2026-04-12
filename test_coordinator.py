# test_coordinator.py
# run with: python3 test_coordinator.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system.coordinator import Coordinator


if __name__ == '__main__':

    print("=" * 60)
    print("Test 1: slow switching")
    print("=" * 60)

    coord = Coordinator(
        n_sessions     = 80,
        mean_duration  = 15,
        std_duration   = 4,
        session_length = 60.0,
        n_buyers       = 10,
        seed           = 42,
        enable_meta    = True,
        enable_risk    = True,
        hmm_warmup     = 21,
    )

    results = coord.run()
    summary = coord.summary()

    print(f"\nSummary (slow switching):")
    print(f"  HMM accuracy : {summary['hmm_accuracy']:.1%}")
    print(f"  Total PnL    : {summary['total_pnl']}")
    print(f"  Sharpe ratio : {summary['sharpe']:.4f}")
    print(f"  Vetoes       : {summary['n_vetoes']}")
    print(f"  Retrains     : {summary['n_retrains']}")

    print(f"\nFirst 25 sessions:")
    print(f"{'Sess':<6} {'True':<18} {'Predicted':<18} "
          f"{'Conf':<8} {'PnL':<10} {'Veto':<6} {'OK'}")
    print("-" * 72)

    for r in results[:25]:
        print(f"{r['session_idx']:<6} {r['true_regime']:<18} "
              f"{r['predicted_regime']:<18} "
              f"{r['confidence']:<8.3f} "
              f"{r['session_pnl']:<10.1f} "
              f"{'Y' if r['veto'] else '':<6} "
              f"{'✓' if r['correct'] else '✗'}")

    print()
    print("=" * 60)
    print("Test 2: fast switching")
    print("=" * 60)

    coord2 = Coordinator(
        n_sessions     = 80,
        mean_duration  = 2,
        std_duration   = 1,
        session_length = 60.0,
        n_buyers       = 10,
        seed           = 42,
        enable_meta    = True,
        enable_risk    = True,
        hmm_warmup     = 21,
    )

    results2 = coord2.run()
    summary2 = coord2.summary()

    print(f"\nSummary (fast switching):")
    print(f"  HMM accuracy : {summary2['hmm_accuracy']:.1%}")
    print(f"  Total PnL    : {summary2['total_pnl']}")
    print(f"  Sharpe ratio : {summary2['sharpe']:.4f}")
    print(f"  Vetoes       : {summary2['n_vetoes']}")
    print(f"  Retrains     : {summary2['n_retrains']}")

    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  Slow switching accuracy : {summary['hmm_accuracy']:.1%}")
    print(f"  Fast switching accuracy : {summary2['hmm_accuracy']:.1%}")
    print(f"  Accuracy drop           : "
          f"{summary2['hmm_accuracy'] - summary['hmm_accuracy']:+.1%}")
    print(f"  Slow switching PnL      : {summary['total_pnl']}")
    print(f"  Fast switching PnL      : {summary2['total_pnl']}")