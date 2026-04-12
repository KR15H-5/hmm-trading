# risk/risk_manager.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RiskManager:
    """
    Global veto controller for the trading system.

    Monitors three risk conditions and halts all agent trading
    when any threshold is breached. Re-enables trading once
    conditions recover and the cooldown period has elapsed.

    WHY A RISK MANAGER?
    -------------------
    Without risk controls, agents can accumulate large losing positions
    during regime transitions when the HMM is uncertain. The risk manager
    provides a safety layer that limits downside while the HMM recovers.

    It also makes the system more realistic — real trading systems always
    have risk limits that override individual strategy signals.

    For the research question, the risk manager's intervention count is
    itself a useful metric: more interventions at fast switching speeds
    indicates the system is less stable under those conditions.

    Parameters
    ----------
    confidence_threshold : HMM confidence below this → veto
                           e.g. 0.45 means "less than 45% sure → stop"
    volatility_threshold : rolling price volatility above this → veto
                           measured as std dev of log-returns
    max_drawdown         : if total PnL drops this far below peak → veto
                           in price units e.g. -100 means stop if down 100
    cooldown             : sessions to wait after veto before re-enabling
    enabled              : if False, risk manager never vetoes
                           used to test system without risk controls
    """

    def __init__(self, confidence_threshold=0.45, volatility_threshold=0.08,
                 max_drawdown=-100.0, cooldown=3, enabled=True):

        self.confidence_threshold = confidence_threshold
        self.volatility_threshold = volatility_threshold
        self.max_drawdown         = max_drawdown
        self.cooldown             = cooldown
        self.enabled              = enabled

        # current veto state
        self.veto_active          = False
        self.sessions_since_veto  = 0

        # peak PnL seen so far — drawdown measured from this
        self.peak_pnl             = 0.0

        # log of all veto events for experiment analysis
        # each entry: {'session': int, 'reason': str, 'value': float}
        self.veto_log             = []

        # total sessions assessed
        self.n_sessions           = 0

    def assess(self, confidence, volatility, current_pnl, session_idx):
        """
        Assess market conditions and decide whether to veto trading.

        Called by the coordinator after every session, before agents
        are activated for the next session.

        Parameters
        ----------
        confidence   : float — HMM posterior probability of predicted regime
        volatility   : float — std dev of log-returns from this session's tape
        current_pnl  : float — total PnL across all agents this episode
        session_idx  : int   — current session number (for logging)

        Returns
        -------
        dict with:
            'veto'      : bool — True means halt all trading
            'reason'    : str  — why the veto was triggered (or 'none')
            'value'     : float — the value that triggered the veto
        """
        self.n_sessions += 1

        if not self.enabled:
            return {'veto': False, 'reason': 'none', 'value': 0.0}

        # ── update peak PnL for drawdown calculation ──────────────────
        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl

        drawdown = current_pnl - self.peak_pnl

        # ── handle cooldown after a veto ──────────────────────────────
        if self.veto_active:
            self.sessions_since_veto += 1
            if self.sessions_since_veto >= self.cooldown:
                # cooldown elapsed — check if conditions have recovered
                if self._conditions_safe(confidence, volatility, drawdown):
                    self.veto_active         = False
                    self.sessions_since_veto = 0
                    return {'veto': False, 'reason': 'recovered', 'value': 0.0}
            # still in cooldown — veto stays active
            return {'veto': True, 'reason': 'cooldown', 'value': self.sessions_since_veto}

        # ── check each risk condition in priority order ───────────────

        # condition 1: low HMM confidence
        if confidence < self.confidence_threshold:
            self._trigger_veto(session_idx, 'low_confidence', confidence)
            return {'veto': True, 'reason': 'low_confidence', 'value': confidence}

        # condition 2: high volatility
        if volatility > self.volatility_threshold:
            self._trigger_veto(session_idx, 'high_volatility', volatility)
            return {'veto': True, 'reason': 'high_volatility', 'value': volatility}

        # condition 3: drawdown exceeded
        if drawdown < self.max_drawdown:
            self._trigger_veto(session_idx, 'drawdown', drawdown)
            return {'veto': True, 'reason': 'drawdown', 'value': drawdown}

        # all conditions safe
        return {'veto': False, 'reason': 'none', 'value': 0.0}

    def _trigger_veto(self, session_idx, reason, value):
        """Record a veto event and activate veto state."""
        self.veto_active         = True
        self.sessions_since_veto = 0
        self.veto_log.append({
            'session': session_idx,
            'reason':  reason,
            'value':   value,
        })
        print(f'[RiskManager] Veto at session {session_idx}: '
              f'{reason} = {value:.4f}')

    def _conditions_safe(self, confidence, volatility, drawdown):
        """
        Check whether all conditions have recovered enough to resume.

        We use slightly relaxed thresholds for recovery to prevent
        rapid veto/resume cycling at the boundary.
        Recovery thresholds are 10% more lenient than trigger thresholds.
        """
        conf_ok = confidence  >= self.confidence_threshold * 1.1
        vol_ok  = volatility  <= self.volatility_threshold * 0.9
        dd_ok   = drawdown    >= self.max_drawdown         * 0.9
        return conf_ok and vol_ok and dd_ok

    def n_vetoes(self):
        """Total number of veto events triggered."""
        return len(self.veto_log)

    def veto_rate(self):
        """Fraction of sessions where trading was halted."""
        if self.n_sessions == 0:
            return 0.0
        n_halted = sum(1 for _ in self.veto_log)
        return n_halted / self.n_sessions

    def reset(self):
        """Reset for a new episode."""
        self.veto_active         = False
        self.sessions_since_veto = 0
        self.peak_pnl            = 0.0
        self.veto_log.clear()
        self.n_sessions          = 0