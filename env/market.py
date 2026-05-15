import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE

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

    sched = REGIME_SCHEDULES[regime]

    s_lo = int(sched['supply'][0] + drift_offset)
    s_hi = int(sched['supply'][1] + drift_offset)
    d_lo = int(sched['demand'][0] + drift_offset)
    d_hi = int(sched['demand'][1] + drift_offset)

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
    start_time = 0.0
    end_time   = session_length

    supply_schedule, demand_schedule = build_schedule(
        regime, start_time, end_time, drift_offset
    )

    order_schedule = {
        'sup':      supply_schedule,
        'dem':      demand_schedule,
        'interval': 5.0,           
        'timemode': 'drip-poisson',  
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
        'dump_tape':     False,
    }

    random.seed(session_idx * 1000)

    sess_id = f'session_{session_idx:04d}'

    result = BSE.market_session(
        sess_id,
        start_time,
        end_time,
        trader_spec,
        order_schedule,
        dump_flags,
        False,         
    )
    tape   = result['exchange'].tape
    trades = [t for t in tape if t['type'] == 'Trade']

    return {
        'tape':    tape,
        'trades':  trades,
        'traders': result['traders'],
        'regime':  regime,
    }