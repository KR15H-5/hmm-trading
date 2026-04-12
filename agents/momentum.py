# agents/momentum.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from agents.base import BaseAgent


class MomentumAgent(BaseAgent):
    """
    Momentum trader — active in the trending regime.

    Strategy: moving average crossover.
        fast MA > slow MA → prices rising → submit a bid (buy)
        fast MA < slow MA → prices falling → submit an ask (sell)

    Parameters (passed via params dict)
    ------------------------------------
    fast_window  : number of recent prices for the fast MA (default 3)
    slow_window  : number of recent prices for the slow MA (default 8)
    max_inventory: maximum units to hold in either direction (default 5)
    """

    def __init__(self, tid, balance, params, time):
        super().__init__(tid, balance, params, time)

        # unpack parameters with sensible defaults
        self.fast_window   = params.get('fast_window',   3)
        self.slow_window   = params.get('slow_window',   8)
        self.max_inventory = params.get('max_inventory', 5)

        # the signal computed in respond() and read in getorder()
        self.signal = None    # 'buy', 'sell', or None

    def _compute_signal(self):
        """
        Compute the moving average crossover signal.

        Returns 'buy', 'sell', or None if we don't have enough prices yet.

        We need at least slow_window prices before we can compute
        a meaningful signal. Before that we return None — no trade.
        """
        if len(self.prices_seen) < self.slow_window:
            return None

        fast_prices = self.prices_seen[-self.fast_window:]
        slow_prices = self.prices_seen[-self.slow_window:]

        fast_ma = sum(fast_prices) / len(fast_prices)
        slow_ma = sum(slow_prices) / len(slow_prices)

        if fast_ma > slow_ma:
            return 'buy'
        elif fast_ma < slow_ma:
            return 'sell'
        else:
            return None    # flat — no signal

    def getorder(self, time, countdown, lob):
        """
        BSE calls this every timestep.

        If we are not active (wrong regime) — return None.
        If we have no signal yet — return None.
        If we are at max inventory — don't add more in same direction.
        Otherwise submit a bid or ask.
        """

        # not our regime — sit out
        if not self.active:
            return None

        # need a signal to trade
        if self.signal is None:
            return None

        market = self.observe(lob)

        if self.signal == 'buy':
            # don't buy if already at max long position
            if self.inventory >= self.max_inventory:
                return None

            # submit a bid just above the best bid to get priority
            # if no best bid exists, use mid price
            if market['best_bid'] is not None:
                price = market['best_bid'] + 1
            elif market['mid'] is not None:
                price = int(market['mid'])
            else:
                return None

            order = BSE.Order(
                self.tid,   # trader ID
                'Bid',      # order type
                price,      # limit price
                1,          # quantity — always 1 in BSE
                time,       # timestamp
                lob['QID'], # quote ID from exchange
            )
            self.lastquote = order
            return order

        elif self.signal == 'sell':
            # don't sell if already at max short position
            if self.inventory <= -self.max_inventory:
                return None

            # submit an ask just below the best ask to get priority
            if market['best_ask'] is not None:
                price = market['best_ask'] - 1
            elif market['mid'] is not None:
                price = int(market['mid'])
            else:
                return None

            order = BSE.Order(
                self.tid,
                'Ask',
                price,
                1,
                time,
                lob['QID'],
            )
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        """
        BSE calls this after every order is processed.

        We do two things:
          1. Call the base class to record any new trade prices
          2. Recompute our signal so getorder() has fresh info next timestep
        """
        # let base class record trade prices into self.prices_seen
        super().respond(time, lob, trade, vrbs)

        # update PnL if we were a party to this trade
        if trade is not None:
            self.update_pnl(trade, self.tid)

        # recompute signal with latest prices
        self.signal = self._compute_signal()