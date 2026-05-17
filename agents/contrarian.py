import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from agents.base import BaseAgent


class ContrarianAgent(BaseAgent):

    def __init__(self, tid, balance, params, time):
        super().__init__(tid, balance, params, time)
        self.zscore_window = params.get('zscore_window', 15)
        self.entry_threshold = params.get('entry_threshold', 1.5)
        self.exit_threshold = params.get('exit_threshold', 0.3)
        self.max_inventory = params.get('max_inventory', 5)
        self.zscore = 0.0
        self.signal = None

    def compute_zscore(self):
        if len(self.prices_seen) < self.zscore_window:
            return 0.0

        window = self.prices_seen[-self.zscore_window:]
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / (len(window) - 1)
        std = math.sqrt(variance)

        if std == 0:
            return 0.0

        return (self.prices_seen[-1] - mean) / std

    def compute_signal(self):
        z = self.zscore

        if self.inventory > 0 and z > -self.exit_threshold:
            return 'close_long'
        if self.inventory < 0 and z < self.exit_threshold:
            return 'close_short'
        if z > self.entry_threshold:
            return 'sell'
        if z < -self.entry_threshold:
            return 'buy'
        return None

    def getorder(self, time, countdown, lob):
        if not self.active or self.signal is None:
            return None

        market = self.observe(lob)

        if self.signal in ('buy', 'close_short'):
            if self.signal == 'buy' and self.inventory >= self.max_inventory:
                return None
            if market['best_ask'] is not None:
                price = market['best_ask']
            elif market['mid'] is not None:
                price = int(market['mid']) + 1
            else:
                return None
            order = BSE.Order(self.tid, 'Bid', price, 1, time, lob['QID'])
            self.lastquote = order
            return order

        if self.signal in ('sell', 'close_long'):
            if self.signal == 'sell' and self.inventory <= -self.max_inventory:
                return None
            if market['best_bid'] is not None:
                price = market['best_bid']
            elif market['mid'] is not None:
                price = int(market['mid']) - 1
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
        self.zscore = self.compute_zscore()
        self.signal = self.compute_signal()
