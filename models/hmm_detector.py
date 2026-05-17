import warnings
warnings.filterwarnings('ignore')
import sys
import os
import math
import numpy as np
from hmmlearn import hmm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIMES = ['trending', 'mean_reverting', 'volatile']


def extract_features(trades):
    if len(trades) < 4:
        return [0.0, 0.0, 0.0, 0.0]

    prices = [t['price'] for t in trades]
    n = len(prices)

    first_half = prices[:n // 2]
    second_half = prices[n // 2:]
    mean_first = sum(first_half) / len(first_half)
    mean_second = sum(second_half) / len(second_half)
    mean_price = sum(prices) / n
    momentum = (mean_second - mean_first) / mean_price if mean_price > 0 else 0.0

    log_returns = []
    for i in range(1, n):
        if prices[i - 1] > 0:
            log_returns.append(math.log(prices[i] / prices[i - 1]))

    if len(log_returns) > 1:
        mean_r = sum(log_returns) / len(log_returns)
        var = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        vol = math.sqrt(var)
    else:
        vol = 0.0

    price_range = (max(prices) - min(prices)) / mean_price if mean_price > 0 else 0.0

    if len(log_returns) > 2:
        mean_r = sum(log_returns) / len(log_returns)
        demeaned = [r - mean_r for r in log_returns]
        num = sum(demeaned[i] * demeaned[i - 1] for i in range(1, len(demeaned)))
        den = sum(d ** 2 for d in demeaned)
        autocorr = num / den if den > 0 else 0.0
    else:
        autocorr = 0.0

    return [momentum, vol, price_range, autocorr]


class HMMDetector:

    def __init__(self, n_states=3, n_iter=100, warmup=10):
        self.n_states = n_states
        self.n_iter = n_iter
        self.warmup = warmup
        self.is_trained = False
        self.model = None
        self.state_to_regime = {}
        self.observation_history = []
        self.alpha = None

    def add_observation(self, features):
        self.observation_history.append(features)

    def train(self):
        if len(self.observation_history) < self.warmup:
            return False

        X = np.array(self.observation_history, dtype=float)
        lengths = [len(X)]

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type='diag',
            n_iter=self.n_iter,
            random_state=42,
            init_params='mc',
        )

        n = self.n_states
        self.model.transmat_ = np.full((n, n), 1.0 / n)
        self.model.startprob_ = np.full(n, 1.0 / n)

        try:
            self.model.fit(X, lengths)
            self.is_trained = True
            self.map_states_to_regimes()
            self.alpha = np.full(n, 1.0 / n)
            return True
        except Exception as e:
            print(f'HMM training failed: {e}')
            return False

    def predict(self, features):
        if not self.is_trained:
            return {
                'regime': 'mean_reverting',
                'confidence': 1.0 / 3,
                'all_probs': {r: 1.0 / 3 for r in REGIMES},
                'trained': False,
            }

        obs = np.array(features, dtype=float)

        # Point-in-time Gaussian emission: log p(obs | state) for each state
        log_emit = np.zeros(self.n_states)
        for state in range(self.n_states):
            mean_s = self.model.means_[state]
            var = self.model.covars_[state]
            lp = 0.0
            for j in range(len(obs)):
                sigma2 = float(var[j]) if var.ndim == 1 else float(var[j, j])
                sigma2 = max(sigma2, 1e-8)
                lp += -0.5 * math.log(2.0 * math.pi * sigma2)
                lp += -0.5 * ((obs[j] - mean_s[j]) ** 2) / sigma2
            log_emit[state] = lp

        log_emit -= log_emit.max()
        emit = np.exp(log_emit)

        # Forward-filter: propagate belief through transition matrix then gate by emission
        alpha_pred = self.alpha @ self.model.transmat_
        alpha_new = alpha_pred * emit

        total = alpha_new.sum()
        if total > 0:
            alpha_new /= total
        else:
            alpha_new = np.full(self.n_states, 1.0 / self.n_states)

        self.alpha = alpha_new

        predicted_state = int(np.argmax(self.alpha))
        confidence = float(self.alpha[predicted_state])
        predicted_regime = self.state_to_regime.get(predicted_state, 'mean_reverting')

        all_probs = {
            regime: float(self.alpha[state])
            for state, regime in self.state_to_regime.items()
        }

        return {
            'regime': predicted_regime,
            'confidence': confidence,
            'all_probs': all_probs,
            'trained': True,
        }

    def update(self, features, window=50):
        self.add_observation(features)
        if len(self.observation_history) > window:
            self.observation_history = self.observation_history[-window:]
        return self.train()

    def map_states_to_regimes(self):
        means = self.model.means_
        volatile_idx = int(np.argmax(means[:, 1]))
        remaining = [i for i in range(self.n_states) if i != volatile_idx]
        trending_idx = remaining[int(np.argmax([abs(means[i, 0]) for i in remaining]))]
        mean_rev_idx = [i for i in remaining if i != trending_idx][0]

        self.state_to_regime = {
            volatile_idx: 'volatile',
            trending_idx: 'trending',
            mean_rev_idx: 'mean_reverting',
        }

        print(f'HMM state mapping: {self.state_to_regime}')
        print(f'Emission means:\n{means}')
