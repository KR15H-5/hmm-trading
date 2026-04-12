# test_agents.py
# run with: python3 test_agents.py

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import BSE
from agents.base import BaseAgent
from agents.momentum import MomentumAgent
from agents.contrarian import ContrarianAgent
from agents.market_maker import MarketMakerAgent

# ─────────────────────────────────────────────────────────────────────────────
# Regime schedules and build_schedule — copied from test_market.py
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
# Run a session with an agent injected live into BSE
# ─────────────────────────────────────────────────────────────────────────────

def run_session_with_live_agent(regime, agent, agent_type, session_idx,
                                 session_length=60.0):
    """
    Run one BSE session with our agent injected as a live proptrader.

    BSE will call agent.getorder() and agent.respond() directly
    every timestep — exactly like any other trader.

    agent_type must be 'MOMENTUM', 'CONTRARIAN', or 'MARKETMAKER'
    — these match the new cases we added to BSE's trader_type().
    """
    start_time = 0.0
    end_time   = session_length

    supply_schedule, demand_schedule = build_schedule(
        regime, start_time, end_time
    )
    order_schedule = {
        'sup':      supply_schedule,
        'dem':      demand_schedule,
        'interval': 5.0,
        'timemode': 'drip-fixed',
    }

    # background traders
    trader_spec = {
        'buyers':  [('ZIC', 5), ('ZIP', 5)],
        'sellers': [('ZIC', 5), ('ZIP', 5)],
        # inject our agent as a proptrader
        # BSE will call trader_type('MOMENTUM', 'P00', {'agent_object': agent})
        # which now returns our pre-built agent object directly
        'proptraders': [(agent_type, 1, {'agent_object': agent})],
    }

    dump_flags = {
        'dump_blotters': False, 'dump_lobs':    False,
        'dump_strats':   False, 'dump_avgbals': False,
        'dump_tape':     True,
    }

    random.seed(session_idx * 1000)

    BSE.market_session(
        f'session_{session_idx:04d}',
        start_time, end_time,
        trader_spec, order_schedule,
        dump_flags, False,
    )

    # read tape
    tape_file = f'session_{session_idx:04d}_tape.csv'
    trades = []
    if os.path.exists(tape_file):
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

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 55)
    print("Test 1: MomentumAgent LIVE in trending regime")
    print("=" * 55)

    momentum = MomentumAgent(
        tid     = 'P00',
        balance = 0,
        params  = {'fast_window': 3, 'slow_window': 8, 'max_inventory': 5},
        time    = 0,
    )
    momentum.active = True

    trades = run_session_with_live_agent(
        'trending', momentum, 'MOMENTUM', session_idx=0
    )

    print(f"Total market trades : {len(trades)}")
    print(f"Agent n_trades      : {momentum.n_trades}")
    print(f"Agent inventory     : {momentum.inventory}")
    print(f"Agent PnL           : {momentum.pnl}")
    print(f"Agent balance       : {momentum.balance}")
    print(f"Agent signal        : {momentum.signal}")
    print(f"Prices seen         : {len(momentum.prices_seen)}")

    print()
    print("=" * 55)
    print("Test 2: ContrarianAgent LIVE in mean-reverting regime")
    print("=" * 55)

    contrarian = ContrarianAgent(
        tid     = 'P00',
        balance = 0,
        params  = {'zscore_window': 15, 'entry_threshold': 1.5,
                   'exit_threshold': 0.3, 'max_inventory': 5},
        time    = 0,
    )
    contrarian.active = True

    trades2 = run_session_with_live_agent(
        'mean_reverting', contrarian, 'CONTRARIAN', session_idx=1
    )

    print(f"Total market trades : {len(trades2)}")
    print(f"Agent n_trades      : {contrarian.n_trades}")
    print(f"Agent inventory     : {contrarian.inventory}")
    print(f"Agent PnL           : {contrarian.pnl}")
    print(f"Agent z-score       : {contrarian.zscore:.4f}")
    print(f"Agent signal        : {contrarian.signal}")

    print()
    print("=" * 55)
    print("Test 3: MarketMakerAgent LIVE in volatile regime")
    print("=" * 55)

    mm = MarketMakerAgent(
        tid     = 'P00',
        balance = 0,
        params  = {'base_spread': 2, 'vol_multiplier': 5,
                   'max_inventory': 8, 'vol_window': 10},
        time    = 0,
    )
    mm.active = True

    trades3 = run_session_with_live_agent(
        'volatile', mm, 'MARKETMAKER', session_idx=2
    )

    print(f"Total market trades : {len(trades3)}")
    print(f"Agent n_trades      : {mm.n_trades}")
    print(f"Agent inventory     : {mm.inventory}")
    print(f"Agent PnL           : {mm.pnl}")
    print(f"Agent spread used   : {mm.current_spread}")

    print()
    print("=" * 55)
    print("Test 4: Agent inactive — should not trade")
    print("=" * 55)

    inactive = MomentumAgent(
        tid     = 'P00',
        balance = 0,
        params  = {'fast_window': 3, 'slow_window': 8, 'max_inventory': 5},
        time    = 0,
    )
    inactive.active = False

    trades4 = run_session_with_live_agent(
        'trending', inactive, 'MOMENTUM', session_idx=3
    )

    print(f"Total market trades : {len(trades4)}")
    print(f"Agent n_trades      : {inactive.n_trades}")
    print(f"Correct (0 trades)  : {inactive.n_trades == 0}")