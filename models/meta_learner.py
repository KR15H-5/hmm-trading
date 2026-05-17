import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hmm_detector import HMMDetector


class MetaLearner:
    def __init__(self, detector, error_window=20, retrain_threshold=0.35,
                 cooldown=10, retrain_window=40, enabled=True):
        self.detector = detector
        self.error_window = error_window
        self.retrain_threshold = retrain_threshold
        self.cooldown = cooldown
        self.retrain_window = retrain_window
        self.enabled = enabled
        self.prediction_record = []
        self.sessions_since_retrain = 0
        self.retrain_log = []
        self.n_sessions = 0

    def record(self, predicted_regime, true_regime, features):
        self.n_sessions += 1
        self.sessions_since_retrain += 1

        correct = (predicted_regime == true_regime)
        self.prediction_record.append(correct)

        if len(self.prediction_record) > self.error_window:
            self.prediction_record = self.prediction_record[-self.error_window:]

        error_rate = self.compute_error_rate()

        retrained = False
        if self.enabled:
            retrained = self.maybe_retrain(error_rate, features)

        return {
            'correct': correct,
            'error_rate': error_rate,
            'retrained': retrained,
        }

    def compute_error_rate(self):
        if len(self.prediction_record) < self.error_window:
            return 0.0
        n_wrong = sum(1 for correct in self.prediction_record if not correct)
        return n_wrong / len(self.prediction_record)

    def maybe_retrain(self, error_rate, features):
        if self.sessions_since_retrain < self.cooldown:
            return False
        if error_rate <= self.retrain_threshold:
            return False

        retrained = self.detector.update(features, window=self.retrain_window)

        if retrained:
            self.sessions_since_retrain = 0
            self.retrain_log.append({
                'session': self.n_sessions,
                'error_rate': error_rate,
            })
            print(f'[MetaLearner] Retrained at session {self.n_sessions} '
                  f'(error_rate={error_rate:.2f})')

        return retrained

    def current_error_rate(self):
        return self.compute_error_rate()

    def n_retrains(self):
        return len(self.retrain_log)

    def accuracy(self):
        if self.n_sessions == 0:
            return 0.0
        n_correct = sum(1 for c in self.prediction_record if c)
        return n_correct / len(self.prediction_record)
