import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import BSE
from models.hmm_detector import extract_features, HMMDetector

REGIME_SCHEDULES = {
    'trending':       {'supply': (82,100), 'demand': (95,120),  'stepmode': 'jittered'},
    'mean_reverting': {'supply': (80,100), 'demand': (100,120), 'stepmode': 'jittered'},
    'volatile':       {'supply': (60,100), 'demand': (100,140), 'stepmode': 'random'},
}

def run_session(regime, idx):
    sched = REGIME_SCHEDULES[regime]
    supply = [{'from':0,'to':300,'ranges':[(sched['supply'][0],sched['supply'][1])],'stepmode':sched['stepmode']}]
    demand = [{'from':0,'to':300,'ranges':[(sched['demand'][0],sched['demand'][1])],'stepmode':sched['stepmode']}]
    order_schedule = {'sup':supply,'dem':demand,'interval':5.0,'timemode':'drip-fixed'}
    trader_spec = {
        'buyers':  [('SHVR',4),('ZIP',3),('ZIC',3)],
        'sellers': [('SHVR',4),('ZIP',3),('ZIC',3)],
        'proptraders': [],
    }
    dump_flags = {'dump_blotters':False,'dump_lobs':False,'dump_strats':False,'dump_avgbals':False,'dump_tape':True}
    random.seed(idx * 777)
    BSE.market_session(f'feat_{idx:04d}', 0.0, 300.0, trader_spec, order_schedule, dump_flags, False)
    tape = f'feat_{idx:04d}_tape.csv'
    trades = []
    with open(tape) as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0].strip() == 'TRD':
                trades.append({'type':'Trade','time':float(parts[1]),'price':int(parts[2])})
    os.remove(tape)
    return trades

if __name__ == '__main__':
    import math

    print(f"{'Regime':<18} {'momentum':>10} {'vol':>10} {'range':>10} {'autocorr':>10}  n_trades")
    print("-" * 72)

    all_features = {'trending': [], 'mean_reverting': [], 'volatile': []}

    for regime in ['trending', 'mean_reverting', 'volatile']:
        for i in range(10):
            trades = run_session(regime, i + {'trending':0,'mean_reverting':100,'volatile':200}[regime])
            f = extract_features(trades)
            all_features[regime].append(f)
            print(f"{regime:<18} {f[0]:>10.4f} {f[1]:>10.4f} {f[2]:>10.4f} {f[3]:>10.4f}  {len(trades)}")
        print()

    print("=== REGIME MEANS ===")
    print(f"{'Regime':<18} {'momentum':>10} {'vol':>10} {'range':>10} {'autocorr':>10}")
    print("-" * 58)
    for regime in ['trending', 'mean_reverting', 'volatile']:
        feats = all_features[regime]
        means = [sum(f[i] for f in feats)/len(feats) for i in range(4)]
        print(f"{regime:<18} {means[0]:>10.4f} {means[1]:>10.4f} {means[2]:>10.4f} {means[3]:>10.4f}")

    print()
    print("=== SEPARABILITY CHECK ===")
    print("For good HMM accuracy you need:")
    print("  volatile:      vol >> 0.05,  range >> 0.3")
    print("  trending:      momentum clearly non-zero, autocorr > 0")
    print("  mean_rev:      autocorr < 0, momentum near zero")
