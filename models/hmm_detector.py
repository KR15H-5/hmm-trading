# models/hmm_detector.py
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import math
import numpy as np
from hmmlearn import hmm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# The three regimes we classify into
REGIMES = ['trending', 'mean_reverting', 'volatile']


def extract_features(trades):
    """
    Convert a session's trade list into a 3-number observation vector.

    Features:
        1. price_momentum   — mean price in second half minus first half
                              positive = trending up, near zero = flat/oscillating
                              REPLACES mean_return which was too noisy

        2. volatility       — std dev of log-returns
                              high = volatile, low = trending/mean-reverting

        3. price_range_ratio — (max - min) / mean
                              wide = volatile, narrow = calm

    WHY REPLACE MEAN_RETURN WITH PRICE_MOMENTUM?
    ---------------------------------------------
    mean_return was the average log-return across all trades in a session.
    Over 40 trades the positive and negative returns cancel out and the
    mean is near zero in all regimes — not useful for discrimination.

    price_momentum compares the first half of the session to the second
    half. In a trending session prices move consistently in one direction
    so the second half average is meaningfully higher or lower than the
    first half. In mean-reverting sessions prices oscillate so both
    halves average to roughly the same value.
    """
    if len(trades) < 4:
        return [0.0, 0.0, 0.0]

    prices = [t['price'] for t in trades]
    n      = len(prices)

    # ── Feature 1: price momentum ─────────────────────────────────────
    first_half  = prices[:n // 2]
    second_half = prices[n // 2:]
    mean_first  = sum(first_half)  / len(first_half)
    mean_second = sum(second_half) / len(second_half)

    # normalise by overall mean so it's scale-invariant
    mean_price = sum(prices) / n
    if mean_price > 0:
        momentum = (mean_second - mean_first) / mean_price
    else:
        momentum = 0.0

    # ── Feature 2: volatility ─────────────────────────────────────────
    log_returns = []
    for i in range(1, n):
        if prices[i - 1] > 0:
            log_returns.append(
                math.log(prices[i] / prices[i - 1])
            )

    if len(log_returns) > 1:
        mean_r = sum(log_returns) / len(log_returns)
        var    = sum((r - mean_r) ** 2 for r in log_returns) / (len(log_returns) - 1)
        vol    = math.sqrt(var)
    else:
        vol = 0.0

    # ── Feature 3: price range ratio ─────────────────────────────────
    price_range = (max(prices) - min(prices)) / mean_price if mean_price > 0 else 0.0

    # lag-1 autocorrelation — negative=mean-reverting, positive=trending
    if len(log_returns) > 2:
        mean_r   = sum(log_returns) / len(log_returns)
        demeaned = [r - mean_r for r in log_returns]
        num  = sum(demeaned[i]*demeaned[i-1] for i in range(1, len(demeaned)))
        den  = sum(d**2 for d in demeaned)
        autocorr = num / den if den > 0 else 0.0
    else:
        autocorr = 0.0

    return [momentum, vol, price_range, autocorr]

class HMMDetector:
    """
    Detects market regimes using a Gaussian Hidden Markov Model.

    HOW IT WORKS
    ------------
    A Hidden Markov Model assumes the world is always in one of N hidden
    states (our regimes). At each timestep it emits an observation (our
    3 features). The model learns:

        1. Transition probabilities — how likely is it to switch from
           regime A to regime B between sessions?

        2. Emission probabilities — given we're in regime A, what do
           the 3 features typically look like? (mean and covariance
           of a Gaussian distribution)

    After training, given a sequence of observations, the HMM can infer
    which hidden state (regime) most likely produced each observation.

    WHY GAUSSIAN HMM
    ----------------
    Our features (log-returns, volatility, price range) are continuous
    numbers. A Gaussian HMM models the emission distribution for each
    state as a multivariate Gaussian — defined by a mean vector and
    covariance matrix. This is appropriate because log-returns are
    approximately normally distributed in financial markets.

    STATE MAPPING
    -------------
    hmmlearn assigns state labels 0, 1, 2 arbitrarily — they don't
    automatically correspond to 'trending', 'mean_reverting', 'volatile'.
    After training we map states to regime names by looking at which
    state has the highest volatility feature (that's volatile),
    which has the most positive mean return (that's trending),
    and the remaining one is mean_reverting.

    Parameters
    ----------
    n_states    : number of hidden states (must be 3 for our system)
    n_iter      : EM algorithm iterations per training call
    warmup      : minimum observations before we start predicting
                  (HMM needs enough data to initialise properly)
    """

    def __init__(self, n_states=3, n_iter=50, warmup=10):
        self.n_states    = n_states
        self.n_iter      = n_iter
        self.warmup      = warmup
        self.is_trained  = False

        # The hmmlearn model — we create it fresh each time we train
        self.model = None

        # Maps HMM state index (0,1,2) to regime name
        # e.g. {0: 'volatile', 1: 'trending', 2: 'mean_reverting'}
        self.state_to_regime = {}

        # History of all observations seen so far
        # Each entry is a list of 3 floats from one session
        self.observation_history = []

        # History of predicted regimes (for meta-learner to read)
        self.prediction_history = []

    def add_observation(self, features):
        """
        Record one session's features into history.

        Called after every session regardless of whether we predict.
        The history is what we train the HMM on.

        Parameters
        ----------
        features : list of 3 floats from extract_features()
        """
        self.observation_history.append(features)

    def train(self):
        """
        Train the HMM on all observations seen so far.

        Uses the Baum-Welch algorithm (Expectation-Maximisation) to
        find the model parameters that best explain the observation
        history.

        We need at least warmup observations before training.
        Returns True if training succeeded, False otherwise.

        WHY RETRAIN PERIODICALLY?
        -------------------------
        The HMM learns what the regimes look like from historical data.
        If the market changes (e.g. volatility increases permanently),
        the model trained on old data becomes stale. The meta-learner
        will call train() again when it detects accuracy degrading.
        """
        if len(self.observation_history) < self.warmup:
            return False

        # Convert observation history to numpy array
        # Shape: (n_observations, n_features) = (n_sessions, 3)
        X = np.array(self.observation_history, dtype=float)

        # hmmlearn requires a lengths array telling it how many
        # observations are in each sequence.
        # We treat our entire history as one long sequence.
        lengths = [len(X)]

        # Create and train the Gaussian HMM
        # covariance_type='full' means each state has its own full
        # covariance matrix — more expressive but needs more data
        # covariance_type='diag' is faster and needs less data
        self.model = hmm.GaussianHMM(
            n_components    = self.n_states,
            covariance_type = 'diag',
            n_iter          = self.n_iter,
            random_state    = 42,
            init_params     = 'mc',   # only initialise means and covars
                                      # don't reinitialise transition matrix
        )

        # set a uniform transition matrix — equal probability of staying
        # or switching to any other regime — prevents the HMM from
        # learning "never switch" from sequential training data
        n = self.n_states
        self.model.transmat_ = np.full((n, n), 1.0 / n)
        self.model.startprob_ = np.full(n, 1.0 / n)

        try:
            self.model.fit(X, lengths)
            self.is_trained = True
            # After training, map state indices to regime names
            self._map_states_to_regimes()
            return True
        except Exception as e:
            print(f'HMM training failed: {e}')
            return False

    def _map_states_to_regimes(self):
        """
        After training, figure out which HMM state corresponds to
        which regime by examining the learned emission means.

        MAPPING LOGIC
        -------------
        The emission mean for each state is a 3-element vector:
            [mean_return, volatility, price_range]

        We identify regimes by:
          - volatile      → highest volatility (feature index 1)
          - trending      → highest absolute mean_return (feature index 0)
          - mean_reverting → whatever is left

        This is deterministic and interpretable — important for your
        report because you can explain exactly how the mapping works.
        """
        means = self.model.means_  # shape: (n_states, n_features)

        # Find which state has the highest volatility — that's volatile
        vol_values   = means[:, 1]    # volatility column
        volatile_idx = int(np.argmax(vol_values))

        # Among remaining states, find highest |mean_return| — that's trending
        remaining = [i for i in range(self.n_states) if i != volatile_idx]
        ret_values   = [abs(means[i, 0]) for i in remaining]
        trending_idx = remaining[int(np.argmax(ret_values))]

        # Whatever is left is mean_reverting
        mean_rev_idx = [i for i in remaining if i != trending_idx][0]

        self.state_to_regime = {
            volatile_idx : 'volatile',
            trending_idx : 'trending',
            mean_rev_idx : 'mean_reverting',
        }

        print(f'HMM state mapping: {self.state_to_regime}')
        print(f'Emission means:\n{means}')
    def predict(self, features):
        """
        Predict regime for a new observation.

        We score the observation directly against each state's learned
        Gaussian emission distribution rather than running the full
        forward algorithm on the entire history.

        WHY THIS APPROACH?
        ------------------
        Using predict_proba on the full history sequence causes the HMM
        to smooth predictions toward the most common historical state —
        it essentially ignores the current observation and predicts the
        base rate. This is called "posterior collapse" and it happens
        when the sequence is long relative to the number of transitions.

        Instead we compute the likelihood of the current observation
        under each state's Gaussian emission distribution directly.
        This gives us a clean per-observation classification without
        temporal smoothing bias.

        P(observation | state) ∝ N(observation; mean_state, cov_state)

        Then we normalise across states to get probabilities.

        Returns
        -------
        dict with regime, confidence, all_probs, trained
        """
        if not self.is_trained:
            default_probs = {r: 1/3 for r in REGIMES}
            return {
                'regime':     'mean_reverting',
                'confidence': 1/3,
                'all_probs':  default_probs,
                'trained':    False,
            }

        obs = np.array(features, dtype=float)

        # compute log-likelihood of this observation under each state's
        # Gaussian emission distribution
        log_likelihoods = []
        for state in range(self.n_states):
            mean = self.model.means_[state]
            # for diagonal covariance, variance is the diagonal
            var  = self.model.covars_[state]  # shape depends on cov type

            # compute log probability under multivariate Gaussian
            # for diagonal covariance this is sum of univariate log probs
            log_prob = 0.0
            for j in range(len(obs)):
                sigma2 = float(var[j]) if var.ndim == 1 else float(var[j, j])
                sigma2 = max(sigma2, 1e-6)  # prevent division by zero
                log_prob += -0.5 * math.log(2 * math.pi * sigma2)
                log_prob += -0.5 * ((obs[j] - mean[j]) ** 2) / sigma2

            log_likelihoods.append(log_prob)

        # convert log-likelihoods to probabilities via softmax
        # subtract max for numerical stability before exponentiating
        max_ll   = max(log_likelihoods)
        exp_ll   = [math.exp(ll - max_ll) for ll in log_likelihoods]
        total    = sum(exp_ll)
        probs    = [e / total for e in exp_ll]

        predicted_state  = int(np.argmax(probs))
        confidence       = float(probs[predicted_state])
        predicted_regime = self.state_to_regime.get(predicted_state, 'mean_reverting')

        all_probs = {
            regime: float(probs[state])
            for state, regime in self.state_to_regime.items()
        }

        self.prediction_history.append(predicted_regime)

        return {
            'regime':     predicted_regime,
            'confidence': confidence,
            'all_probs':  all_probs,
            'trained':    True,
        }
    def update(self, features, window=50):
        """
        Add a new observation and optionally retrain on a rolling window.

        Called by the meta-learner when it decides retraining is needed.

        Parameters
        ----------
        features : list of 3 floats — the new session's features
        window   : how many recent observations to train on
                   older data is discarded — this is the rolling window

        Returns
        -------
        bool — True if retraining happened, False if not enough data

        WHY A ROLLING WINDOW?
        ---------------------
        If we always trained on all historical data, the model would be
        heavily influenced by old market conditions. A rolling window
        means the HMM only learns from recent behaviour.

        For example if the market has been volatile for 30 sessions but
        was trending for the 50 sessions before that, a rolling window
        of 40 would only see the volatile period and learn it well.
        A full history model would be confused by the mix.

        This is the core of what makes it "online" learning — the model
        continuously adapts as new data arrives rather than being fixed
        after an initial training phase.

        The tradeoff: too small a window and the model is noisy and
        unstable. Too large and it's slow to adapt. 50 sessions is a
        reasonable default for our experiment timescale.
        """

        # add the new observation to history
        self.add_observation(features)

        # trim history to rolling window
        if len(self.observation_history) > window:
            self.observation_history = self.observation_history[-window:]

        # retrain on the trimmed window
        return self.train()