# models/meta_learner.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hmm_detector import HMMDetector


class MetaLearner:
    """
    Monitors HMM classification accuracy and triggers retraining
    when performance degrades.

    WHY DO WE NEED THIS?
    --------------------
    A static HMM trained once at the start will gradually become stale.
    As regime switching accelerates, the distribution of observations
    shifts and the model's learned emission means no longer match reality.

    The meta-learner detects this by tracking a rolling error rate —
    the fraction of recent sessions where the HMM predicted incorrectly.
    When this exceeds a threshold it triggers retraining on recent data.

    This directly addresses the research question: "to what extent does
    online meta-learning mitigate accuracy degradation?" By comparing
    runs WITH and WITHOUT the meta-learner we can measure its benefit.

    HOW IT WORKS
    ------------
    After every session:
      1. Record whether the HMM prediction matched the true regime
      2. Compute rolling error rate over the last error_window sessions
      3. If error rate > retrain_threshold AND cooldown has elapsed:
           - trigger HMM retraining on a rolling window of recent data
           - record the retraining event for analysis

    Parameters
    ----------
    detector         : the HMMDetector instance to monitor
    error_window     : number of recent sessions to compute error rate over
    retrain_threshold: error rate above which retraining is triggered
    cooldown         : minimum sessions between retraining events
                       prevents thrashing — retraining every session
                       would be expensive and destabilising
    retrain_window   : how many recent observations to train on
    enabled          : if False, meta-learner never triggers retraining
                       used for the ablation condition in experiments
    """

    def __init__(self, detector, error_window=20, retrain_threshold=0.35,
                 cooldown=10, retrain_window=40, enabled=True):

        self.detector          = detector
        self.error_window      = error_window
        self.retrain_threshold = retrain_threshold
        self.cooldown          = cooldown
        self.retrain_window    = retrain_window
        self.enabled           = enabled

        # rolling record of correct/incorrect predictions
        # True = correct, False = incorrect
        self.prediction_record = []

        # how many sessions since the last retraining
        self.sessions_since_retrain = 0

        # log of all retraining events for analysis
        # each entry: {'session': int, 'error_rate': float}
        self.retrain_log = []

        # total number of sessions observed
        self.n_sessions = 0

    def record(self, predicted_regime, true_regime, features):
        """
        Record one session's outcome and decide whether to retrain.

        Called after every session by the coordinator.

        Parameters
        ----------
        predicted_regime : what the HMM predicted
        true_regime      : what the regime actually was (ground truth)
        features         : the 3-number feature vector for this session
                           passed to HMM update if retraining is triggered

        Returns
        -------
        dict with:
            'correct'     : bool — was this prediction right?
            'error_rate'  : float — rolling error rate over error_window
            'retrained'   : bool — did we retrain this session?
        """

        self.n_sessions += 1
        self.sessions_since_retrain += 1

        # record whether this prediction was correct
        correct = (predicted_regime == true_regime)
        self.prediction_record.append(correct)

        # keep only the last error_window records
        if len(self.prediction_record) > self.error_window:
            self.prediction_record = self.prediction_record[-self.error_window:]

        # compute rolling error rate
        error_rate = self._compute_error_rate()

        # decide whether to retrain
        retrained = False
        if self.enabled:
            retrained = self._maybe_retrain(error_rate, features)

        return {
            'correct':    correct,
            'error_rate': error_rate,
            'retrained':  retrained,
        }

    def _compute_error_rate(self):
        """
        Rolling error rate = fraction of recent predictions that were wrong.

        Returns 0.0 if we have fewer than error_window observations —
        we don't want to trigger retraining prematurely on noisy early data.
        """
        if len(self.prediction_record) < self.error_window:
            return 0.0

        n_wrong = sum(1 for correct in self.prediction_record if not correct)
        return n_wrong / len(self.prediction_record)

    def _maybe_retrain(self, error_rate, features):
        """
        Trigger retraining if conditions are met.

        Conditions:
          1. Error rate exceeds retrain_threshold
          2. Cooldown period has elapsed since last retraining
          3. HMM has enough history to retrain on

        Returns True if retraining was triggered.
        """
        # check cooldown
        if self.sessions_since_retrain < self.cooldown:
            return False

        # check error threshold
        if error_rate <= self.retrain_threshold:
            return False

        # trigger retraining
        retrained = self.detector.update(features, window=self.retrain_window)

        if retrained:
            self.sessions_since_retrain = 0
            self.retrain_log.append({
                'session':    self.n_sessions,
                'error_rate': error_rate,
            })
            print(f'[MetaLearner] Retrained at session {self.n_sessions} '
                  f'(error_rate={error_rate:.2f})')

        return retrained

    def current_error_rate(self):
        """Return the current rolling error rate."""
        return self._compute_error_rate()

    def n_retrains(self):
        """How many times has retraining been triggered?"""
        return len(self.retrain_log)

    def accuracy(self):
        """
        Overall accuracy across all sessions observed.
        Used in experiment evaluation.
        """
        if self.n_sessions == 0:
            return 0.0
        n_correct = sum(1 for c in self.prediction_record if c)
        return n_correct / len(self.prediction_record)