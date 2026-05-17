import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE
from agents.base import BaseAgent


class MarketMakerAgent(BaseAgent):

    def __init__(self, tid, balance, params, time):
        super().__init__(tid, balance, params, time)
        self.base_spread = params.get('base_spread', 2)
        self.vol_multiplier = params.get('vol_multiplier', 5)
        self.max_inventory = params.get('max_inventory', 8)
        self.vol_window = params.get('vol_window', 10)
        self.current_spread = self.base_spread
        self.current_mid = None

    def compute_volatility(self):
        if len(self.prices_seen) < 2:
            return 0.0
        window = self.prices_seen[-self.vol_window:]
        mean = sum(window) / len(window)
        variance = sum((p - mean) ** 2 for p in window) / (len(window) - 1)
        return math.sqrt(variance)

    def compute_spread(self):
        vol = self.compute_volatility()
        spread = self.base_spread + self.vol_multiplier * vol
        return max(self.base_spread, int(round(spread)))

    def getorder(self, time, countdown, lob):
        if not self.active:
            return None

        market = self.observe(lob)
        if market['mid'] is None:
            return None

        self.current_mid = market['mid']
        self.current_spread = self.compute_spread()

        bid_price = max(1, int(self.current_mid - self.current_spread))
        ask_price = min(500, int(self.current_mid + self.current_spread))

        if self.inventory >= self.max_inventory:
            order = BSE.Order(self.tid, 'Ask', ask_price, 1, time, lob['QID'])
            self.lastquote = order
            return order
        if self.inventory <= -self.max_inventory:
            order = BSE.Order(self.tid, 'Bid', bid_price, 1, time, lob['QID'])
            self.lastquote = order
            return order

        if int(time * 100) % 2 == 0:
            order = BSE.Order(self.tid, 'Bid', bid_price, 1, time, lob['QID'])
        else:
            order = BSE.Order(self.tid, 'Ask', ask_price, 1, time, lob['QID'])

        self.lastquote = order
        return order

    def respond(self, time, lob, trade, vrbs):
        super().respond(time, lob, trade, vrbs)
        if trade is not None:
            self.update_pnl(trade, self.tid)
        self.current_spread = self.compute_spread()
