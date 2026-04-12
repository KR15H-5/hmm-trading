# agents/contrarian.py

import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from agents.base import BaseAgent


class ContrarianAgent(BaseAgent):
    """
    Contrarian trader — active in the mean-reverting regime.

    Strategy: z-score mean reversion.
        z > +entry_threshold  → price too high → sell (expect it to fall)
        z < -entry_threshold  → price too low  → buy  (expect it to rise)
        |z| < exit_threshold  → price back to normal → close position

    Parameters (passed via params dict)
    ------------------------------------
    zscore_window    : how many recent prices to compute mean/std over (default 15)
    entry_threshold  : z-score level to enter a trade (default 1.5)
    exit_threshold   : z-score level to exit a trade  (default 0.3)
    max_inventory    : maximum units to hold in either direction (default 5)
    """

    def __init__(self, tid, balance, params, time):
        super().__init__(tid, balance, params, time)

        self.zscore_window   = params.get('zscore_window',   15)
        self.entry_threshold = params.get('entry_threshold', 1.5)
        self.exit_threshold  = params.get('exit_threshold',  0.3)
        self.max_inventory   = params.get('max_inventory',   5)

        self.zscore = 0.0     # current z-score, updated in respond()
        self.signal = None    # 'buy', 'sell', 'close', or None

    def _compute_zscore(self):
        """
        Compute z-score of the most recent price relative to
        the rolling window of recent prices.

        Returns 0.0 if we don't have enough prices yet.

        WHY Z-SCORE?
        ------------
        Raw price alone tells us nothing — a price of 110 might be
        high or normal depending on recent history.
        Z-score normalises by recent mean and std so we always know
        exactly how unusual the current price is relative to recent
        behaviour. That's what mean-reversion trading is built on.
        """
        if len(self.prices_seen) < self.zscore_window:
            return 0.0

        window = self.prices_seen[-self.zscore_window:]
        mean   = sum(window) / len(window)

        variance = sum((p - mean) ** 2 for p in window) / (len(window) - 1)
        std      = math.sqrt(variance)

        # avoid division by zero if all prices are identical
        if std == 0:
            return 0.0

        current_price = self.prices_seen[-1]
        return (current_price - mean) / std

    def _compute_signal(self):
        """
        Translate z-score into a trading signal.

        ENTRY:
            z > +entry_threshold → price unusually high → sell
            z < -entry_threshold → price unusually low  → buy

        EXIT:
            if we're already long and z > -exit_threshold → close (sell)
            if we're already short and z < +exit_threshold → close (buy)

        WHY SEPARATE ENTRY AND EXIT THRESHOLDS?
        ----------------------------------------
        We enter when the price is clearly extreme (z=1.5).
        We exit earlier, when it's returned close to normal (z=0.3).
        If we waited for z=0 to exit we'd often give back our profits
        as the price overshoots in the other direction.
        """
        z = self.zscore

        # exit existing long position — price has recovered
        if self.inventory > 0 and z > -self.exit_threshold:
            return 'close_long'

        # exit existing short position — price has recovered
        if self.inventory < 0 and z < self.exit_threshold:
            return 'close_short'

        # enter new position if price is extreme
        if z > self.entry_threshold:
            return 'sell'    # price too high, expect fall

        if z < -self.entry_threshold:
            return 'buy'     # price too low, expect rise

        return None

    def getorder(self, time, countdown, lob):
        """
        BSE calls this every timestep.
        """

        if not self.active:
            return None

        if self.signal is None:
            return None

        market = self.observe(lob)

        # ── buying signals ────────────────────────────────────────────────
        if self.signal in ('buy', 'close_short'):

            if self.signal == 'buy' and self.inventory >= self.max_inventory:
                return None

            if market['best_ask'] is not None:
                price = market['best_ask']
            elif market['mid'] is not None:
                price = int(market['mid']) + 1
            else:
                return None

            order = BSE.Order(
                self.tid, 'Bid', price, 1, time, lob['QID']
            )
            self.lastquote = order
            return order

        # ── selling signals ───────────────────────────────────────────────
        if self.signal in ('sell', 'close_long'):

            if self.signal == 'sell' and self.inventory <= -self.max_inventory:
                return None

            if market['best_bid'] is not None:
                price = market['best_bid']
            elif market['mid'] is not None:
                price = int(market['mid']) - 1
            else:
                return None

            order = BSE.Order(
                self.tid, 'Ask', price, 1, time, lob['QID']
            )
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        BSE calls this after every order is processed.
        Update prices seen, PnL, z-score and signal.
        """
        super().respond(time, lob, trade, vrbs)

        if trade is not None:
            self.update_pnl(trade, self.tid)

        # recompute z-score and signal with latest prices
        self.zscore = self._compute_zscore()
        self.signal = self._compute_signal()