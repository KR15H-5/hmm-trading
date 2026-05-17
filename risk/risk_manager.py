import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RiskManager:

    def __init__(self, confidence_threshold=0.45, volatility_threshold=0.08,
                 max_drawdown=-100.0, cooldown=3, enabled=True):
        self.confidence_threshold = confidence_threshold
        self.volatility_threshold = volatility_threshold
        self.max_drawdown = max_drawdown
        self.cooldown = cooldown
        self.enabled = enabled
        self.veto_active = False
        self.sessions_since_veto = 0
        self.peak_pnl = 0.0
        self.veto_log = []
        self.n_sessions = 0

    def assess(self, confidence, volatility, current_pnl, session_idx):
        self.n_sessions += 1

        if not self.enabled:
            return {'veto': False, 'reason': 'none', 'value': 0.0}

        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl

        drawdown = current_pnl - self.peak_pnl

        if self.veto_active:
            self.sessions_since_veto += 1
            if self.sessions_since_veto >= self.cooldown:
                if self.conditions_safe(confidence, volatility, drawdown):
                    self.veto_active = False
                    self.sessions_since_veto = 0
                    return {'veto': False, 'reason': 'recovered', 'value': 0.0}
            return {'veto': True, 'reason': 'cooldown', 'value': self.sessions_since_veto}

        if confidence < self.confidence_threshold:
            self.trigger_veto(session_idx, 'low_confidence', confidence)
            return {'veto': True, 'reason': 'low_confidence', 'value': confidence}

        if volatility > self.volatility_threshold:
            self.trigger_veto(session_idx, 'high_volatility', volatility)
            return {'veto': True, 'reason': 'high_volatility', 'value': volatility}

        if drawdown < self.max_drawdown:
            self.trigger_veto(session_idx, 'drawdown', drawdown)
            return {'veto': True, 'reason': 'drawdown', 'value': drawdown}

        return {'veto': False, 'reason': 'none', 'value': 0.0}

    def trigger_veto(self, session_idx, reason, value):
        self.veto_active = True
        self.sessions_since_veto = 0
        self.veto_log.append({
            'session': session_idx,
            'reason': reason,
            'value': value,
        })
        print(f'[RiskManager] Veto at session {session_idx}: {reason} = {value:.4f}')

    def conditions_safe(self, confidence, volatility, drawdown):
        # Hysteresis: resume only when metrics clear a tighter bound than the trigger
        conf_ok = confidence >= self.confidence_threshold * 1.1
        vol_ok = volatility <= self.volatility_threshold * 0.9
        dd_ok = drawdown >= self.max_drawdown * 0.9
        return conf_ok and vol_ok and dd_ok

    def n_vetoes(self):
        return len(self.veto_log)

    def veto_rate(self):
        if self.n_sessions == 0:
            return 0.0
        return len(self.veto_log) / self.n_sessions

    def reset(self):
        self.veto_active = False
        self.sessions_since_veto = 0
        self.peak_pnl = 0.0
        self.veto_log.clear()
        self.n_sessions = 0
