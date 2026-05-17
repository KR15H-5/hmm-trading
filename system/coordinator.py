import sys
import os
import random
import math
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from models.hmm_detector import HMMDetector, extract_features
from models.meta_learner import MetaLearner
from risk.risk_manager import RiskManager
from agents.momentum import MomentumAgent
from agents.contrarian import ContrarianAgent
from agents.market_maker import MarketMakerAgent


REGIME_SCHEDULES = {
    'trending': {
        'supply': (70, 90), 'demand': (85, 110),
        'stepmode': 'jittered', 'drift': 1.5,
    },
    'mean_reverting': {
        'supply': (80, 100), 'demand': (100, 120),
        'stepmode': 'jittered', 'drift': 0.0,
    },
    'volatile': {
        'supply': (40, 100), 'demand': (100, 160),
        'stepmode': 'random', 'drift': 0.0,
    },
}

REGIMES = ['trending', 'mean_reverting', 'volatile']
AGENT_TYPE_MAP = {
    'trending': 'MOMENTUM',
    'mean_reverting': 'CONTRARIAN',
    'volatile': 'MARKETMAKER',
}


def build_schedule(regime, start_time, end_time, drift_offset=0.0):
    sched = REGIME_SCHEDULES[regime]
    s_lo = max(1, int(sched['supply'][0] + drift_offset))
    s_hi = max(1, min(500, int(sched['supply'][1] + drift_offset)))
    d_lo = max(1, int(sched['demand'][0] + drift_offset))
    d_hi = min(500, int(sched['demand'][1] + drift_offset))
    mid = (start_time + end_time) / 2.0

    if regime == 'trending':
        shift = 15
        supply = [
            {'from': start_time, 'to': mid,
             'ranges': [(s_lo, s_hi)], 'stepmode': sched['stepmode']},
            {'from': mid, 'to': end_time,
             'ranges': [(min(499, s_lo + shift), min(500, s_hi + shift))],
             'stepmode': sched['stepmode']},
        ]
        demand = [
            {'from': start_time, 'to': mid,
             'ranges': [(d_lo, d_hi)], 'stepmode': sched['stepmode']},
            {'from': mid, 'to': end_time,
             'ranges': [(min(499, d_lo + shift), min(500, d_hi + shift))],
             'stepmode': sched['stepmode']},
        ]
    else:
        supply = [{'from': start_time, 'to': end_time,
                   'ranges': [(s_lo, s_hi)], 'stepmode': sched['stepmode']}]
        demand = [{'from': start_time, 'to': end_time,
                   'ranges': [(d_lo, d_hi)], 'stepmode': sched['stepmode']}]

    return supply, demand


def compute_volatility(trades):
    if len(trades) < 2:
        return 0.0
    prices = [t['price'] for t in trades]
    log_returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))
    if len(log_returns) < 2:
        return 0.0
    mean = sum(log_returns) / len(log_returns)
    var = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(var)


def run_bse_session(true_regime, session_idx, active_agent,
                    agent_type, drift_offset, session_length, n_buyers):
    sup, dem = build_schedule(true_regime, 0.0, session_length, drift_offset)
    order_schedule = {
        'sup': sup, 'dem': dem,
        'interval': 5.0, 'timemode': 'drip-fixed',
    }

    if active_agent is not None and active_agent.active:
        proptraders = [(agent_type, 1, {'agent_object': active_agent})]
    else:
        proptraders = []

    trader_spec = {
        'buyers': [('SHVR', 4), ('ZIP', 3), ('ZIC', 3)],
        'sellers': [('SHVR', 4), ('ZIP', 3), ('ZIC', 3)],
        'proptraders': proptraders,
    }
    dump_flags = {
        'dump_blotters': False, 'dump_lobs': False,
        'dump_strats': False, 'dump_avgbals': False,
        'dump_tape': True,
    }

    random.seed(session_idx * 1000)
    sess_id = f'coord_{session_idx:04d}'
    BSE.market_session(sess_id, 0.0, session_length,
                       trader_spec, order_schedule, dump_flags, False)

    tape_file = sess_id + '_tape.csv'
    trades = []
    if os.path.exists(tape_file):
        with open(tape_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if parts[0].strip() == 'TRD':
                    trades.append({
                        'type': 'Trade',
                        'time': float(parts[1].strip()),
                        'price': int(parts[2].strip()),
                    })
        os.remove(tape_file)
    return trades


def draw_duration(mean_duration, std_duration):
    duration = random.gauss(mean_duration, std_duration)
    return max(1, int(round(duration)))


class RegimeSwitcher:
    def __init__(self, mean_duration, std_duration, seed):
        self.mean_duration = mean_duration
        self.std_duration = std_duration
        self.rng = random.Random(seed)
        self.current = self.rng.choice(REGIMES)
        self.remaining = draw_duration(mean_duration, std_duration)

    def next_regime(self):
        self.remaining -= 1
        if self.remaining <= 0:
            others = [r for r in REGIMES if r != self.current]
            self.current = self.rng.choice(others)
            self.remaining = draw_duration(self.mean_duration, self.std_duration)
        return self.current


class Coordinator:

    def __init__(self, n_sessions=100, mean_duration=10, std_duration=3,
                 session_length=300.0, n_buyers=10, seed=42,
                 enable_meta=True, enable_risk=True, hmm_warmup=21,
                 retrain_threshold=0.60):
        self.n_sessions = n_sessions
        self.session_length = session_length
        self.n_buyers = n_buyers
        self.seed = seed
        self.hmm_warmup = max(3, (hmm_warmup // 3) * 3)

        self.switcher_mean = mean_duration
        self.switcher_std = std_duration
        self.switcher = RegimeSwitcher(mean_duration, std_duration, seed)

        self.detector = HMMDetector(n_states=3, n_iter=100, warmup=self.hmm_warmup)
        self.meta = MetaLearner(
            detector=self.detector,
            error_window=15,
            retrain_threshold=retrain_threshold,
            cooldown=10,
            retrain_window=40,
            enabled=enable_meta,
        )
        self.risk = RiskManager(
            confidence_threshold=0.45,
            volatility_threshold=0.15,
            max_drawdown=-200.0,
            cooldown=3,
            enabled=enable_risk,
        )

        self.agents = {
            'trending': MomentumAgent(
                'P00', 0,
                {'fast_window': 3, 'slow_window': 8, 'max_inventory': 5},
                0
            ),
            'mean_reverting': ContrarianAgent(
                'P00', 0,
                {'zscore_window': 15, 'entry_threshold': 1.5,
                 'exit_threshold': 0.3, 'max_inventory': 5},
                0
            ),
            'volatile': MarketMakerAgent(
                'P00', 0,
                {'base_spread': 2, 'vol_multiplier': 5,
                 'max_inventory': 8, 'vol_window': 10},
                0
            ),
        }

        self.drift_offset = 0.0
        self.results = []

        sessions_per_regime = self.hmm_warmup // 3
        warmup_seq = REGIMES * sessions_per_regime
        rng_warmup = random.Random(seed + 1)
        rng_warmup.shuffle(warmup_seq)
        self.warmup_sequence = warmup_seq

    def set_active_agent(self, regime):
        for r, agent in self.agents.items():
            agent.active = (r == regime)

    def deactivate_all_agents(self):
        for agent in self.agents.values():
            agent.active = False

    def total_pnl(self):
        return sum(agent.pnl for agent in self.agents.values())

    def last_live_volatility(self):
        live = [r for r in self.results if r['veto_reason'] != 'warmup']
        if not live:
            return 0.0
        return live[-1]['volatility']

    def run(self):
        self.results = []
        prev_total_pnl = 0.0
        predicted_regime = 'mean_reverting'
        confidence = 1 / 3
        prev_true_regime = None

        for session_idx in range(self.n_sessions):

            if session_idx < self.hmm_warmup:
                warmup_regime = self.warmup_sequence[session_idx]

                if warmup_regime == 'trending':
                    self.drift_offset += REGIME_SCHEDULES['trending']['drift']
                else:
                    self.drift_offset *= 0.9

                trades = run_bse_session(
                    true_regime=warmup_regime,
                    session_idx=session_idx,
                    active_agent=None,
                    agent_type=None,
                    drift_offset=self.drift_offset,
                    session_length=self.session_length,
                    n_buyers=self.n_buyers,
                )
                features = extract_features(trades)
                volatility = compute_volatility(trades)

                self.detector.add_observation(features)

                if session_idx == self.hmm_warmup - 1:
                    self.detector.train()
                    result = self.detector.predict(features)
                    predicted_regime = result['regime']
                    confidence = result['confidence']
                    self.switcher = RegimeSwitcher(
                        self.switcher_mean, self.switcher_std, self.seed + 999
                    )

                self.results.append({
                    'session_idx': session_idx,
                    'true_regime': warmup_regime,
                    'predicted_regime': predicted_regime,
                    'confidence': round(confidence, 4),
                    'correct': predicted_regime == warmup_regime,
                    'veto': False,
                    'veto_reason': 'warmup',
                    'retrained': False,
                    'error_rate': 0.0,
                    'session_pnl': 0.0,
                    'session_pnl_correct': 0.0,
                    'session_pnl_incorrect': 0.0,
                    'total_pnl': 0.0,
                    'n_trades': len(trades),
                    'volatility': round(volatility, 6),
                    'drift_offset': round(self.drift_offset, 4),
                    'is_transition': False,
                    'transition_from': None,
                    'transition_to': None,
                })
                prev_true_regime = warmup_regime
                continue

            true_regime = self.switcher.next_regime()

            is_transition = (prev_true_regime is not None and
                             true_regime != prev_true_regime)
            transition_from = prev_true_regime if is_transition else None
            transition_to = true_regime if is_transition else None

            if true_regime == 'trending':
                self.drift_offset += REGIME_SCHEDULES['trending']['drift']
            else:
                self.drift_offset *= 0.9

            risk_result = self.risk.assess(
                confidence=confidence,
                volatility=self.last_live_volatility(),
                current_pnl=self.total_pnl(),
                session_idx=session_idx,
            )

            if risk_result['veto']:
                self.deactivate_all_agents()
                active_agent = None
                agent_type = None
            else:
                self.set_active_agent(predicted_regime)
                active_agent = self.agents[predicted_regime]
                agent_type = AGENT_TYPE_MAP[predicted_regime]

            trades = run_bse_session(
                true_regime=true_regime,
                session_idx=session_idx,
                active_agent=active_agent,
                agent_type=agent_type,
                drift_offset=self.drift_offset,
                session_length=self.session_length,
                n_buyers=self.n_buyers,
            )

            features = extract_features(trades)
            volatility = compute_volatility(trades)

            if risk_result['veto']:
                self.detector.add_observation(features)

            if self.detector.is_trained:
                hmm_result = self.detector.predict(features)
                predicted_regime = hmm_result['regime']
                confidence = hmm_result['confidence']

            meta_result = self.meta.record(
                predicted_regime=predicted_regime,
                true_regime=true_regime,
                features=features,
            )

            current_total_pnl = self.total_pnl()
            session_pnl = current_total_pnl - prev_total_pnl
            prev_total_pnl = current_total_pnl

            is_correct = (predicted_regime == true_regime)
            session_pnl_correct = session_pnl if is_correct else 0.0
            session_pnl_incorrect = session_pnl if not is_correct else 0.0

            self.results.append({
                'session_idx': session_idx,
                'true_regime': true_regime,
                'predicted_regime': predicted_regime,
                'confidence': round(confidence, 4),
                'correct': is_correct,
                'veto': risk_result['veto'],
                'veto_reason': risk_result['reason'],
                'retrained': meta_result['retrained'],
                'error_rate': round(meta_result['error_rate'], 4),
                'session_pnl': round(session_pnl, 2),
                'session_pnl_correct': round(session_pnl_correct, 2),
                'session_pnl_incorrect': round(session_pnl_incorrect, 2),
                'total_pnl': round(current_total_pnl, 2),
                'n_trades': len(trades),
                'volatility': round(volatility, 6),
                'drift_offset': round(self.drift_offset, 4),
                'is_transition': is_transition,
                'transition_from': transition_from,
                'transition_to': transition_to,
            })

            prev_true_regime = true_regime

        return self.results

    def summary(self):
        if not self.results:
            return {}

        live = [r for r in self.results if r['veto_reason'] != 'warmup']
        if not live:
            return {}

        n = len(live)
        n_correct = sum(1 for r in live if r['correct'])
        n_vetoes = sum(1 for r in live if r['veto'])
        total_pnl = live[-1]['total_pnl']

        per_regime_acc = {}
        for regime in REGIMES:
            regime_sessions = [r for r in live if r['true_regime'] == regime]
            if regime_sessions:
                correct = sum(1 for r in regime_sessions if r['correct'])
                per_regime_acc[regime] = round(correct / len(regime_sessions), 4)
            else:
                per_regime_acc[regime] = 0.0

        prev_regime = live[0]['true_regime']
        in_lag = False
        lag_count = 0
        trans_key = None
        trans_lag_data = {}

        for r in live[1:]:
            if r['true_regime'] != prev_regime:
                in_lag = True
                lag_count = 0
                trans_key = f"{prev_regime}_to_{r['true_regime']}"
                if trans_key not in trans_lag_data:
                    trans_lag_data[trans_key] = []
            if in_lag:
                lag_count += 1
                if r['predicted_regime'] == r['true_regime']:
                    trans_lag_data[trans_key].append(lag_count)
                    in_lag = False
            prev_regime = r['true_regime']

        per_transition_lag = {
            k: round(sum(v) / len(v), 3) for k, v in trans_lag_data.items() if v
        }

        pnl_when_correct = sum(r['session_pnl_correct'] for r in live)
        pnl_when_incorrect = sum(r['session_pnl_incorrect'] for r in live)
        n_correct_sessions = sum(1 for r in live if r['correct'] and not r['veto'])
        n_incorrect_sessions = sum(1 for r in live if not r['correct'] and not r['veto'])
        avg_pnl_correct = pnl_when_correct / n_correct_sessions if n_correct_sessions > 0 else 0.0
        avg_pnl_incorrect = pnl_when_incorrect / n_incorrect_sessions if n_incorrect_sessions > 0 else 0.0

        trans_conf = [r['confidence'] for r in live if r['is_transition']]
        stable_conf = [r['confidence'] for r in live if not r['is_transition']]
        mean_conf_transition = sum(trans_conf) / len(trans_conf) if trans_conf else 0.0
        mean_conf_stable = sum(stable_conf) / len(stable_conf) if stable_conf else 0.0

        pnls = [r['session_pnl'] for r in live]
        mean_pnl = sum(pnls) / n
        if n > 1:
            var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / (n - 1)
            std_pnl = math.sqrt(var_pnl)
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        return {
            'n_sessions': n,
            'hmm_accuracy': round(n_correct / n, 4),
            'total_pnl': round(total_pnl, 2),
            'sharpe': round(sharpe, 4),
            'n_vetoes': n_vetoes,
            'n_retrains': self.meta.n_retrains(),
            'acc_trending': per_regime_acc.get('trending', 0.0),
            'acc_mean_reverting': per_regime_acc.get('mean_reverting', 0.0),
            'acc_volatile': per_regime_acc.get('volatile', 0.0),
            'per_transition_lag': per_transition_lag,
            'pnl_when_correct': round(pnl_when_correct, 2),
            'pnl_when_incorrect': round(pnl_when_incorrect, 2),
            'avg_pnl_correct': round(avg_pnl_correct, 2),
            'avg_pnl_incorrect': round(avg_pnl_incorrect, 2),
            'conf_at_transition': round(mean_conf_transition, 4),
            'conf_at_stable': round(mean_conf_stable, 4),
        }
