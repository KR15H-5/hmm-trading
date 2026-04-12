# test_risk_manager.py
# run with: python3 test_risk_manager.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from risk.risk_manager import RiskManager


if __name__ == '__main__':

    rm = RiskManager(
        confidence_threshold = 0.45,
        volatility_threshold = 0.08,
        max_drawdown         = -50.0,
        cooldown             = 3,
        enabled              = True,
    )

    print("=" * 55)
    print("Test 1: normal conditions — no veto")
    print("=" * 55)

    result = rm.assess(
        confidence   = 0.80,   # HMM is confident
        volatility   = 0.03,   # low volatility
        current_pnl  = 10.0,   # slightly profitable
        session_idx  = 0,
    )
    print(f"Veto: {result['veto']}  Reason: {result['reason']}")
    assert result['veto'] == False

    print()
    print("=" * 55)
    print("Test 2: low confidence — should veto")
    print("=" * 55)

    result = rm.assess(
        confidence   = 0.35,   # HMM uncertain
        volatility   = 0.03,
        current_pnl  = 10.0,
        session_idx  = 1,
    )
    print(f"Veto: {result['veto']}  Reason: {result['reason']}")
    assert result['veto'] == True
    assert result['reason'] == 'low_confidence'

    print()
    print("=" * 55)
    print("Test 3: cooldown — veto stays active for 3 sessions")
    print("=" * 55)

    for i in range(3):
        result = rm.assess(
            confidence   = 0.90,   # conditions recovered
            volatility   = 0.01,
            current_pnl  = 10.0,
            session_idx  = 2 + i,
        )
        print(f"  Session {2+i}: Veto={result['veto']}  "
              f"Reason={result['reason']}")

    print()
    print("=" * 55)
    print("Test 4: after cooldown — should resume trading")
    print("=" * 55)

    result = rm.assess(
        confidence   = 0.90,
        volatility   = 0.01,
        current_pnl  = 10.0,
        session_idx  = 5,
    )
    print(f"Veto: {result['veto']}  Reason: {result['reason']}")
    assert result['veto'] == False

    print()
    print("=" * 55)
    print("Test 5: high volatility — should veto")
    print("=" * 55)

    result = rm.assess(
        confidence   = 0.85,
        volatility   = 0.12,   # spike
        current_pnl  = 10.0,
        session_idx  = 6,
    )
    print(f"Veto: {result['veto']}  Reason: {result['reason']}")
    assert result['veto'] == True
    assert result['reason'] == 'high_volatility'

    # reset for drawdown test
    rm2 = RiskManager(max_drawdown=-50.0, cooldown=3, enabled=True)

    print()
    print("=" * 55)
    print("Test 6: drawdown — should veto")
    print("=" * 55)

    # first build up a peak
    rm2.assess(confidence=0.8, volatility=0.02,
               current_pnl=100.0, session_idx=0)

    # then drop hard
    result = rm2.assess(
        confidence   = 0.8,
        volatility   = 0.02,
        current_pnl  = 40.0,   # dropped 60 from peak of 100
        session_idx  = 1,
    )
    print(f"Veto: {result['veto']}  Reason: {result['reason']}")
    print(f"Drawdown: {40.0 - 100.0} vs threshold: {rm2.max_drawdown}")
    assert result['veto'] == True
    assert result['reason'] == 'drawdown'

    print()
    print("=" * 55)
    print("Test 7: disabled risk manager — never vetoes")
    print("=" * 55)

    rm3 = RiskManager(enabled=False)
    result = rm3.assess(
        confidence   = 0.10,   # terrible confidence
        volatility   = 0.50,   # extreme volatility
        current_pnl  = -500.0, # massive loss
        session_idx  = 0,
    )
    print(f"Veto: {result['veto']}  (should be False)")
    assert result['veto'] == False

    print()
    print(f"Total vetoes in rm  : {rm.n_vetoes()}")
    print(f"Total vetoes in rm2 : {rm2.n_vetoes()}")
    print()
    print("All risk manager tests passed.")