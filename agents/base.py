import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE

class BaseAgent(BSE.Trader):
    def __init__(self, tid, balance, params, time):
        super().__init__(self.__class__.__name__, tid, balance, params, time)
        self.inventory = 0
        self.pnl = 0.0
        self.active = True
        self.prices_seen = []

    def observe(self, lob):
        best_bid = lob['bids']['best']
        best_ask = lob['asks']['best']

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            mid = float(best_bid)
        elif best_ask is not None:
            mid = float(best_ask)
        else:
            mid = None

        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'mid': mid,
        }

    def update_pnl(self, trade, my_tid):
        if 'party1' not in trade or 'party2' not in trade:
            return

        price = trade['price']

        if trade['party1'] == my_tid or trade['party2'] == my_tid:
            if self.orders and self.orders[0].otype == 'Bid':
                self.inventory += 1
                self.pnl -= price
            else:
                self.inventory -= 1
                self.pnl += price

    def bookkeep(self, time, trade, order, vrbs):
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        price = trade['price']

        if trade.get('party1') == self.tid:
            if self.lastquote is not None and self.lastquote.otype == 'Bid':
                self.inventory += 1
                self.pnl -= price
                self.balance -= price
            else:
                self.inventory -= 1
                self.pnl += price
                self.balance += price

        elif trade.get('party2') == self.tid:
            if self.lastquote is not None and self.lastquote.otype == 'Bid':
                self.inventory += 1
                self.pnl -= price
                self.balance -= price
            else:
                self.inventory -= 1
                self.pnl += price
                self.balance += price

        self.n_trades += 1
        self.orders = []
        self.n_quotes = 0

    def getorder(self, time, countdown, lob):
        return None

    def respond(self, time, lob, trade, vrbs):
        if trade is not None:
            self.prices_seen.append(trade['price'])
            self.prices_seen = self.prices_seen[-50:]
