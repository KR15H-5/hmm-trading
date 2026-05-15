import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from agents.base import BaseAgent


class MomentumAgent(BaseAgent):
    def __init__(self, tid, balance, params, time):
        super().__init__(tid, balance, params, time)

        self.fast_window   = params.get('fast_window',   3)
        self.slow_window   = params.get('slow_window',   8)
        self.max_inventory = params.get('max_inventory', 5)

        self.signal = None

    def _compute_signal(self):
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
            return None

    def getorder(self, time, countdown, lob):
        if not self.active:
            return None

        if self.signal is None:
            return None

        market = self.observe(lob)

        if self.signal == 'buy':
            if self.inventory >= self.max_inventory:
                return None

            if market['best_bid'] is not None:
                price = market['best_bid'] + 1
            elif market['mid'] is not None:
                price = int(market['mid'])
            else:
                return None

            order = BSE.Order(self.tid, 'Bid', price, 1, time, lob['QID'])
            self.lastquote = order
            return order

        elif self.signal == 'sell':
            if self.inventory <= -self.max_inventory:
                return None

            if market['best_ask'] is not None:
                price = market['best_ask'] - 1
            elif market['mid'] is not None:
                price = int(market['mid'])
            else:
                return None

            order = BSE.Order(self.tid, 'Ask', price, 1, time, lob['QID'])
            self.lastquote = order
            return order

        return None

    def respond(self, time, lob, trade, vrbs):
        super().respond(time, lob, trade, vrbs)

        if trade is not None:
            self.update_pnl(trade, self.tid)

        self.signal = self._compute_signal()
