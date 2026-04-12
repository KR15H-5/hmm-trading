# test_market.py
# run with: python3 test_market.py

import sys
import os
import random
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import BSE

# ─────────────────────────────────────────────────────────────────────────────
# Block 2 — Regime schedules
# ─────────────────────────────────────────────────────────────────────────────

REGIME_SCHEDULES = {
    'trending': {
        'supply':   (82, 100),
        'demand':   (95, 120),
        'stepmode': 'jittered',
        'drift':    0.4,
    },
    'mean_reverting': {
        'supply':   (80, 100),
        'demand':   (100, 120),
        'stepmode': 'jittered',
        'drift':    0.0,
    },
    'volatile': {
        'supply':   (60, 100),
        'demand':   (100, 140),
        'stepmode': 'random',
        'drift':    0.0,
    },
}

REGIMES = ['trending', 'mean_reverting', 'volatile']

# ─────────────────────────────────────────────────────────────────────────────
# Block 3 — Build BSE schedule from regime
# ─────────────────────────────────────────────────────────────────────────────

def build_schedule(regime, start_time, end_time, drift_offset=0.0):
    sched = REGIME_SCHEDULES[regime]

    s_lo = max(1,   int(sched['supply'][0] + drift_offset))
    s_hi = max(1,   int(sched['supply'][1] + drift_offset))
    d_lo = max(1,   int(sched['demand'][0] + drift_offset))
    d_hi = min(500, int(sched['demand'][1] + drift_offset))

    supply_schedule = [{'from': start_time, 'to': end_time,
                        'ranges': [(s_lo, s_hi)], 'stepmode': sched['stepmode']}]
    demand_schedule = [{'from': start_time, 'to': end_time,
                        'ranges': [(d_lo, d_hi)], 'stepmode': sched['stepmode']}]

    return supply_schedule, demand_schedule

# ─────────────────────────────────────────────────────────────────────────────
# Block 4 — Run one BSE session
# ─────────────────────────────────────────────────────────────────────────────

def run_session(regime, session_idx, n_buyers=10, n_sellers=10,
                session_length=60.0, drift_offset=0.0, extra_traders=None):

    start_time = 0.0
    end_time   = session_length

    supply_schedule, demand_schedule = build_schedule(
        regime, start_time, end_time, drift_offset
    )

    order_schedule = {
        'sup':      supply_schedule,
        'dem':      demand_schedule,
        'interval': 5.0,
        'timemode': 'drip-fixed',
    }

    n_zic = n_buyers // 2
    n_zip = n_buyers - n_zic

    trader_spec = {
        'buyers':      [('ZIC', n_zic), ('ZIP', n_zip)],
        'sellers':     [('ZIC', n_zic), ('ZIP', n_zip)],
        'proptraders': extra_traders if extra_traders else [],
    }

    dump_flags = {
        'dump_blotters': False,
        'dump_lobs':     False,
        'dump_strats':   False,
        'dump_avgbals':  False,
        'dump_tape':     True,
    }

    random.seed(session_idx * 1000)

    sess_id = f'session_{session_idx:04d}'

    BSE.market_session(
        sess_id,
        start_time,
        end_time,
        trader_spec,
        order_schedule,
        dump_flags,
        False,
    )

    tape_file = sess_id + '_tape.csv'
    trades = []

    with open(tape_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0].strip() == 'TRD':
                trades.append({
                    'type':  'Trade',
                    'time':  float(parts[1].strip()),
                    'price': int(parts[2].strip()),
                })

    os.remove(tape_file)

    return {
        'trades':  trades,
        'regime':  regime,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Block 5 — Run many sessions with regime switching
# ─────────────────────────────────────────────────────────────────────────────

def draw_duration(mean_duration, std_duration):
    """
    How many sessions do we stay in this regime?

    We draw randomly so switching isn't perfectly regular.
    A perfectly regular switch would be easy for the HMM to exploit
    and wouldn't reflect real market behaviour.

    min(1) so we always stay at least one full session.
    """
    duration = random.gauss(mean_duration, std_duration)
    return max(1, int(round(duration)))


def run_episode(n_sessions=20, mean_duration=5, std_duration=2,
                n_buyers=10, n_sellers=10, session_length=60.0,
                seed=42):
    """
    Run a full episode of back-to-back BSE sessions with regime switching.

    Parameters
    ----------
    n_sessions    : total number of sessions to run
    mean_duration : average number of sessions before switching regime
    std_duration  : randomness in how long each regime lasts
    n_buyers      : background buyers per session
    n_sellers     : background sellers per session
    session_length: simulated seconds per session
    seed          : random seed — same seed always gives same episode

    Returns
    -------
    list of dicts, one per session:
        session_idx  : 0, 1, 2, ...
        true_regime  : which regime was actually running
        trades       : list of trade dicts from BSE
        n_trades     : number of trades that happened
        avg_price    : average transaction price that session
        price_range  : max price minus min price that session
        drift_offset : how much cumulative drift has built up
    """

    random.seed(seed)

    results      = []
    drift_offset = 0.0

    # start in a random regime and draw how long to stay
    current_regime     = random.choice(REGIMES)
    sessions_remaining = draw_duration(mean_duration, std_duration)

    for session_idx in range(n_sessions):

        # ── step 1: check if it is time to switch regime ──────────────────
        # we decrement BEFORE running the session so session 0 counts
        sessions_remaining -= 1

        if sessions_remaining <= 0:
            # pick any regime except the current one
            other_regimes      = [r for r in REGIMES if r != current_regime]
            current_regime     = random.choice(other_regimes)
            sessions_remaining = draw_duration(mean_duration, std_duration)

        # ── step 2: update drift ──────────────────────────────────────────
        # drift only builds up during trending sessions
        # it slowly decays when we are in any other regime
        # this mimics how real price trends gradually fade
        if current_regime == 'trending':
            drift_offset += REGIME_SCHEDULES['trending']['drift']
        else:
            drift_offset *= 0.9

        # ── step 3: run the session ───────────────────────────────────────
        session_result = run_session(
            regime         = current_regime,
            session_idx    = session_idx,
            n_buyers       = n_buyers,
            n_sellers      = n_sellers,
            session_length = session_length,
            drift_offset   = drift_offset,
        )

        # ── step 4: summarise the trades ──────────────────────────────────
        trades = session_result['trades']
        prices = [t['price'] for t in trades]

        if len(prices) > 0:
            avg_price   = sum(prices) / len(prices)
            price_range = max(prices) - min(prices)
        else:
            avg_price   = 0.0
            price_range = 0.0

        # ── step 5: store everything ──────────────────────────────────────
        results.append({
            'session_idx':  session_idx,
            'true_regime':  current_regime,
            'trades':       trades,
            'n_trades':     len(trades),
            'avg_price':    round(avg_price, 1),
            'price_range':  price_range,
            'drift_offset': round(drift_offset, 2),
        })

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("Test 1: one session per regime")
    print("=" * 55)

    for regime in ['trending', 'mean_reverting', 'volatile']:
        result = run_session(regime, session_idx=0)
        trades = result['trades']
        prices = [t['price'] for t in trades]

        if len(trades) == 0:
            print(f"\nRegime     : {regime}")
            print(f"Trades     : 0 — something is wrong")
            continue

        print(f"\nRegime     : {regime}")
        print(f"Trades     : {len(trades)}")
        print(f"Min price  : {min(prices)}")
        print(f"Max price  : {max(prices)}")
        print(f"Avg price  : {sum(prices) / len(prices):.1f}")
        print(f"Price range: {max(prices) - min(prices)}")

    print()
    print("=" * 55)
    print("Test 2: 20 sessions with regime switching")
    print("=" * 55)

    results = run_episode(
        n_sessions    = 20,
        mean_duration = 5,
        std_duration  = 2,
        seed          = 42,
    )

    print(f"\n{'Sess':<6} {'Regime':<18} {'Trades':<8} "
          f"{'Avg Price':<12} {'Range':<8} {'Drift'}")
    print("-" * 60)

    for r in results:
        print(f"{r['session_idx']:<6} {r['true_regime']:<18} "
              f"{r['n_trades']:<8} {r['avg_price']:<12} "
              f"{r['price_range']:<8} {r['drift_offset']}")