import sys, os, random, math
sys.path.insert(0, '.')
import BSE
from models.hmm_detector import extract_features

def run_split_session(regime, idx, session_length=300.0):
    mid = session_length / 2.0

    if regime == 'trending':
        supply = [
            {'from': 0.0,  'to': mid,           'ranges': [(70, 90)],   'stepmode': 'jittered'},
            {'from': mid,  'to': session_length, 'ranges': [(85, 105)],  'stepmode': 'jittered'},
        ]
        demand = [
            {'from': 0.0,  'to': mid,           'ranges': [(85, 110)],  'stepmode': 'jittered'},
            {'from': mid,  'to': session_length, 'ranges': [(100, 125)], 'stepmode': 'jittered'},
        ]
    elif regime == 'mean_reverting':
        supply = [{'from': 0.0, 'to': session_length, 'ranges': [(80, 100)],  'stepmode': 'jittered'}]
        demand = [{'from': 0.0, 'to': session_length, 'ranges': [(100, 120)], 'stepmode': 'jittered'}]
    else:
        supply = [{'from': 0.0, 'to': session_length, 'ranges': [(40, 100)],  'stepmode': 'random'}]
        demand = [{'from': 0.0, 'to': session_length, 'ranges': [(100, 160)], 'stepmode': 'random'}]

    order_schedule = {'sup': supply, 'dem': demand, 'interval': 5.0, 'timemode': 'drip-fixed'}
    trader_spec = {
        'buyers':      [('SHVR', 4), ('ZIP', 3), ('ZIC', 3)],
        'sellers':     [('SHVR', 4), ('ZIP', 3), ('ZIC', 3)],
        'proptraders': [],
    }
    dump_flags = {'dump_blotters': False, 'dump_lobs': False,
                  'dump_strats': False, 'dump_avgbals': False, 'dump_tape': True}
    random.seed(idx * 777)
    sess_id = f'split_{idx:04d}'
    BSE.market_session(sess_id, 0.0, session_length, trader_spec, order_schedule, dump_flags, False)
    tape = sess_id + '_tape.csv'
    trades = []
    with open(tape) as f:
        for line in f:
            p = line.strip().split(',')
            if p[0].strip() == 'TRD':
                trades.append({'type': 'Trade', 'time': float(p[1]), 'price': int(p[2])})
    os.remove(tape)
    return trades

if __name__ == '__main__':
    print(f"{'Regime':<18} {'momentum':>10} {'vol':>10} {'range':>10} {'autocorr':>10}  n_trades")
    print("-" * 72)

    all_features = {'trending': [], 'mean_reverting': [], 'volatile': []}
    offsets = {'trending': 0, 'mean_reverting': 100, 'volatile': 200}

    for regime in ['trending', 'mean_reverting', 'volatile']:
        for i in range(10):
            trades = run_split_session(regime, i + offsets[regime])
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
    print("=== SEPARABILITY ===")
    t_mom  = sum(f[0] for f in all_features['trending']) / 10
    mr_mom = sum(f[0] for f in all_features['mean_reverting']) / 10
    print(f"Trending momentum:      {t_mom:+.4f}")
    print(f"Mean-rev momentum:      {mr_mom:+.4f}")
    print(f"Separation:             {abs(t_mom - mr_mom):.4f}  (need > 0.05 to be useful)")
    print(f"Volatile vol:           {sum(f[1] for f in all_features['volatile'])/10:.4f}  (vs trending {sum(f[1] for f in all_features['trending'])/10:.4f})")
