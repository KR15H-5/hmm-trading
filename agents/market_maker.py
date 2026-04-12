# agents/market_maker.py

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from agents.base import BaseAgent


class MarketMakerAgent(BaseAgent):
    """
    Market maker — active in the volatile regime.

    Strategy: continuously quote both sides of the market around mid price.
    Profit comes from buying at the bid and selling at the ask — the spread.

    The spread widens when recent volatility is high to protect against
    adverse price moves while holding inventory.

    Parameters (passed via params dict)
    ------------------------------------
    base_spread      : minimum half-spread in price units (default 2)
                       quotes bid at mid-base_spread, ask at mid+base_spread
    vol_multiplier   : how much to widen spread per unit of volatility (default 5)
    max_inventory    : position limit — if hit, only quote the closing side (default 8)
    vol_window       : number of recent prices to compute volatility from (default 10)
    """

    def __init__(self, tid, balance, params, time):
        super().__init__(tid, balance, params, time)

        self.base_spread    = params.get('base_spread',    2)
        self.vol_multiplier = params.get('vol_multiplier', 5)
        self.max_inventory  = params.get('max_inventory',  8)
        self.vol_window     = params.get('vol_window',     10)

        self.current_spread = self.base_spread
        self.current_mid    = None

    def _compute_volatility(self):
        """
        Compute recent price volatility as std dev of prices.

        We use raw price std dev here rather than log-returns
        because we need it in price units to set the spread.
        Returns 0.0 if not enough prices yet.
        """
        if len(self.prices_seen) < 2:
            return 0.0

        window = self.prices_seen[-self.vol_window:]
        mean   = sum(window) / len(window)

        variance = sum((p - mean) ** 2 for p in window) / (len(window) - 1)
        return math.sqrt(variance)

    def _compute_spread(self):
        """
        Compute the half-spread to quote each side of mid.

        spread = base_spread + vol_multiplier * volatility

        Example:
            base_spread    = 2
            vol_multiplier = 5
            volatility     = 3.0

            spread = 2 + 5 * 3.0 = 17 units each side of mid

        Minimum is always base_spread — we never quote tighter
        than our base even in calm conditions.
        """
        vol    = self._compute_volatility()
        spread = self.base_spread + self.vol_multiplier * vol
        return max(self.base_spread, int(round(spread)))

    def getorder(self, time, countdown, lob):
        """
        BSE calls this every timestep.

        Unlike momentum and contrarian which trade occasionally,
        the market maker tries to always have quotes on both sides.

        Each timestep it either places a bid or an ask — alternating
        between sides so both sides stay fresh.

        INVENTORY SKEW:
        If inventory gets too long (bought too much, can't sell),
        only quote the ask side to reduce position.
        If inventory gets too short (sold too much, can't buy),
        only quote the bid side to reduce position.
        """

        if not self.active:
            return None

        market = self.observe(lob)

        if market['mid'] is None:
            return None

        self.current_mid    = market['mid']
        self.current_spread = self._compute_spread()

        bid_price = int(self.current_mid - self.current_spread)
        ask_price = int(self.current_mid + self.current_spread)

        # clamp to BSE valid price range
        bid_price = max(1,   bid_price)
        ask_price = min(500, ask_price)

        # ── inventory skew ────────────────────────────────────────────────
        # if too long — only sell to reduce position
        if self.inventory >= self.max_inventory:
            order = BSE.Order(
                self.tid, 'Ask', ask_price, 1, time, lob['QID']
            )
            self.lastquote = order
            return order

        # if too short — only buy to reduce position
        if self.inventory <= -self.max_inventory:
            order = BSE.Order(
                self.tid, 'Bid', bid_price, 1, time, lob['QID']
            )
            self.lastquote = order
            return order

        # ── normal operation — alternate bid and ask ──────────────────────
        # use time to alternate: even timesteps quote bid, odd quote ask
        # this keeps both sides of the book refreshed
        if int(time * 100) % 2 == 0:
            order = BSE.Order(
                self.tid, 'Bid', bid_price, 1, time, lob['QID']
            )
        else:
            order = BSE.Order(
                self.tid, 'Ask', ask_price, 1, time, lob['QID']
            )

        self.lastquote = order
        return order

    def respond(self, time, lob, trade, vrbs):
        """
        BSE calls this after every order is processed.
        Update prices seen, PnL and spread.
        """
        super().respond(time, lob, trade, vrbs)

        if trade is not None:
            self.update_pnl(trade, self.tid)

        # recompute spread with latest volatility
        self.current_spread = self._compute_spread()