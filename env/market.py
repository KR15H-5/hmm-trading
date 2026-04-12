# env/market.py

import sys
import os
import random

# Add project root to path so we can import BSE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
# The three regime schedules.
# Each one defines what the BSE market looks like in that regime.
#
# 'supply' : (low, high) — range of seller costs
# 'demand' : (low, high) — range of buyer valuations  
# 'stepmode': how BSE spaces prices within the range
#             'jittered' = evenly spaced + small noise = moderate variance
#             'random'   = fully random draws = high variance
# 'drift'  : price units added to both ranges each session
#             only trending uses this — creates the upward trend over time

REGIME_SCHEDULES = {
    'trending': {
        'supply':   (82, 100),
        'demand':   (100, 120),
        'stepmode': 'jittered',
        'drift':    0.4,
    },
    'mean_reverting': {
        'supply':   (85, 105),
        'demand':   (95, 115),
        'stepmode': 'jittered',
        'drift':    0.0,
    },
    'volatile': {
        'supply':   (75, 110),
        'demand':   (90, 125),
        'stepmode': 'random',
        'drift':    0.0,
    },
}

def build_schedule(regime, start_time, end_time, drift_offset=0.0):
    """
    Convert a regime name into BSE supply/demand schedules.

    Parameters
    ----------
    regime       : 'trending', 'mean_reverting', or 'volatile'
    start_time   : session start (e.g. 0.0)
    end_time     : session end   (e.g. 60.0)
    drift_offset : cumulative drift applied so far (only matters for trending)
                   starts at 0, increases by 0.4 each trending session

    Returns
    -------
    supply_schedule, demand_schedule  — both in BSE list-of-dicts format
    """

    sched = REGIME_SCHEDULES[regime]

    # Apply drift offset to both ranges
    # int() because BSE expects integer prices
    s_lo = int(sched['supply'][0] + drift_offset)
    s_hi = int(sched['supply'][1] + drift_offset)
    d_lo = int(sched['demand'][0] + drift_offset)
    d_hi = int(sched['demand'][1] + drift_offset)

    # Safety check: supply top must stay below demand bottom
    # If drift has pushed them together, force a 2-unit gap
    if s_hi >= d_lo:
        mid = (s_hi + d_lo) // 2
        s_hi = mid - 1
        d_lo = mid + 1

    supply_schedule = [{
        'from':     start_time,
        'to':       end_time,
        'ranges':   [(s_lo, s_hi)],
        'stepmode': sched['stepmode'],
    }]

    demand_schedule = [{
        'from':     start_time,
        'to':       end_time,
        'ranges':   [(d_lo, d_hi)],
        'stepmode': sched['stepmode'],
    }]

    return supply_schedule, demand_schedule

def run_session(regime, session_idx, n_buyers=10, n_sellers=10,
                session_length=60.0, drift_offset=0.0,
                extra_traders=None):
    """
    Run one BSE session with the given regime injected via the schedule.

    Parameters
    ----------
    regime         : 'trending', 'mean_reverting', or 'volatile'
    session_idx    : integer index — used for unique session ID and seeding
    n_buyers       : number of background buyer traders
    n_sellers      : number of background seller traders
    session_length : how long the session runs in simulated seconds
    drift_offset   : cumulative price drift so far (only used in trending)
    extra_traders  : our intelligent agents passed as proptraders to BSE
                     format: [('TYPE', count, params_dict), ...]
                     Leave None for now — we add agents in a later step

    Returns
    -------
    dict with:
        'tape'    : full BSE tape (trades + cancellations)
        'trades'  : filtered list of just the completed trades
        'traders' : dict of tid -> Trader object (has .balance, .n_trades)
        'regime'  : the regime that was injected (ground truth label)
    """

    # -------------------------------------------------------------------------
    # Step 1: build supply/demand schedules for this regime
    # -------------------------------------------------------------------------
    start_time = 0.0
    end_time   = session_length

    supply_schedule, demand_schedule = build_schedule(
        regime, start_time, end_time, drift_offset
    )

    order_schedule = {
        'sup':      supply_schedule,
        'dem':      demand_schedule,
        'interval': 5.0,             # new customer orders arrive ~every 5 seconds
        'timemode': 'drip-poisson',  # Poisson arrivals — realistic, not clockwork
    }

    # -------------------------------------------------------------------------
    # Step 2: define the background trader population
    #
    # We split evenly between ZIC and ZIP.
    # ZIC  = random price within limit, no learning
    # ZIP  = adjusts margin based on market, simple learning
    #
    # This mix is standard in BSE literature (Cliff 1997, 2018) and ensures
    # realistic price discovery without adaptive traders overwhelming our agents.
    # -------------------------------------------------------------------------
    n_zic = n_buyers // 2           # e.g. 5 ZIC buyers
    n_zip = n_buyers - n_zic        # e.g. 5 ZIP buyers

    trader_spec = {
        'buyers':      [('ZIC', n_zic), ('ZIP', n_zip)],
        'sellers':     [('ZIC', n_zic), ('ZIP', n_zip)],
        'proptraders': extra_traders if extra_traders else [],
    }

    # -------------------------------------------------------------------------
    # Step 3: turn off all BSE file output
    #
    # BSE can write blotters, LOB frames, strategy logs, tape CSVs to disk.
    # We don't want any of that — we're running 2400 trials and will collect
    # everything we need from the return value instead.
    # -------------------------------------------------------------------------
    dump_flags = {
        'dump_blotters': False,
        'dump_lobs':     False,
        'dump_strats':   False,
        'dump_avgbals':  False,
        'dump_tape':     False,
    }

    # -------------------------------------------------------------------------
    # Step 4: seed BSE's random state
    #
    # BSE uses Python's global random module internally — it calls
    # random.randint(), random.random() etc directly.
    # We seed it with session_idx so:
    #   - every session produces different but reproducible behaviour
    #   - re-running the same trial with the same seed gives identical results
    # The * 1000 gives enough spacing so adjacent sessions don't have
    # overlapping random sequences.
    # -------------------------------------------------------------------------
    random.seed(session_idx * 1000)

    # -------------------------------------------------------------------------
    # Step 5: run the BSE session
    # -------------------------------------------------------------------------
    sess_id = f'session_{session_idx:04d}'

    result = BSE.market_session(
        sess_id,
        start_time,
        end_time,
        trader_spec,
        order_schedule,
        dump_flags,
        False,          # sess_vrbs=False — silent, no print output
    )

    # -------------------------------------------------------------------------
    # Step 6: extract what we need from BSE's return value
    #
    # result['exchange'].tape contains ALL events: trades AND cancellations
    # We filter to just trades because cancellations have no price information
    # and we don't want them polluting our feature extraction later
    # -------------------------------------------------------------------------
    tape   = result['exchange'].tape
    trades = [t for t in tape if t['type'] == 'Trade']

    return {
        'tape':    tape,
        'trades':  trades,
        'traders': result['traders'],
        'regime':  regime,
    }