# agents/base.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import BSE


class BaseAgent(BSE.Trader):
    """
    Base class for all our intelligent agents.

    Inherits from BSE.Trader so BSE can call getorder() and respond()
    on our agents exactly like any other trader.

    Adds three things BSE.Trader doesn't have:
      1. inventory  — net position (+ = long, - = short)
      2. pnl        — running profit and loss
      3. active     — whether this agent is allowed to trade right now
                      the coordinator sets this based on HMM regime

    Parameters
    ----------
    tid      : trader ID string e.g. 'momentum_01'
    balance  : starting cash balance
    params   : dict of agent-specific parameters
    time     : simulation time at creation
    """

    def __init__(self, tid, balance, params, time):
        # call BSE.Trader's init
        super().__init__(self.__class__.__name__, tid, balance, params, time)

        self.inventory   = 0       # net units held (+ve long, -ve short)
        self.pnl         = 0.0     # cumulative profit and loss
        self.active      = True    # coordinator switches this on/off per regime
        self.prices_seen = []      # recent transaction prices this agent observed

    def observe(self, lob):
        """
        Read useful numbers out of the LOB dict BSE passes in.

        BSE's lob dict:
            lob['bids']['best']  — best bid price (or None)
            lob['asks']['best']  — best ask price (or None)
            lob['bids']['n']     — number of bids on the book
            lob['asks']['n']     — number of asks on the book
            lob['tape']          — list of recent events

        Returns a clean dict so subclasses don't have to
        dig into the lob structure directly.
        """
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

        recent_trades = [
            event['price']
            for event in lob['tape']
            if event['type'] == 'Trade'
        ]

        return {
            'best_bid':      best_bid,
            'best_ask':      best_ask,
            'mid':           mid,
            'recent_trades': recent_trades,
        }

    def update_pnl(self, trade, my_tid):
        """
        Called after a trade to update inventory and PnL.

        If we were the buyer  → inventory goes up,   cash goes down
        If we were the seller → inventory goes down, cash goes up

        Guards against trades that don't have party information
        (e.g. trades read from tape file rather than live from BSE).
        """
        if 'party1' not in trade or 'party2' not in trade:
            return

        price = trade['price']

        if trade['party1'] == my_tid or trade['party2'] == my_tid:
            if self.orders and self.orders[0].otype == 'Bid':
                self.inventory += 1
                self.pnl       -= price
            else:
                self.inventory -= 1
                self.pnl       += price

    def bookkeep(self, time, trade, order, vrbs):
        """
        Override BSE's bookkeep to handle proptrader PnL correctly.

        BSE's default bookkeep assumes the trader has a customer order
        in self.orders[0] to compare against the transaction price.
        Our agents don't get customer orders — they submit their own
        orders directly via getorder().

        So we track PnL ourselves and just record the trade in the
        blotter without trying to compute profit against a customer order.
        """
        # record trade in blotter
        self.blotter.append(trade)
        self.blotter = self.blotter[-self.blotter_length:]

        price = trade['price']

        # work out if we were the buyer or seller
        # party1 = passive side (resting order), party2 = aggressive side
        if trade.get('party1') == self.tid:
            # we were the passive side — check what order type we had
            if self.lastquote is not None and self.lastquote.otype == 'Bid':
                # we were a resting bid that got lifted — we bought
                self.inventory += 1
                self.pnl       -= price
                self.balance   -= price
            else:
                # we were a resting ask that got hit — we sold
                self.inventory -= 1
                self.pnl       += price
                self.balance   += price

        elif trade.get('party2') == self.tid:
            # we were the aggressive side
            if self.lastquote is not None and self.lastquote.otype == 'Bid':
                # we submitted a bid that crossed the ask — we bought
                self.inventory += 1
                self.pnl       -= price
                self.balance   -= price
            else:
                # we submitted an ask that crossed the bid — we sold
                self.inventory -= 1
                self.pnl       += price
                self.balance   += price

        self.n_trades += 1

        # clear our order — BSE expects this after a trade
        self.orders = []
        self.n_quotes = 0

    def getorder(self, time, countdown, lob):
        """
        BSE calls this every timestep asking if we want to submit an order.
        Base class returns None — subclasses override this with their strategy.
        """
        return None

    def respond(self, time, lob, trade, vrbs):
        """
        BSE calls this after every order is processed.
        Base class records recent trade prices.
        Subclasses override to add strategy-specific logic.
        """
        if trade is not None:
            self.prices_seen.append(trade['price'])
            self.prices_seen = self.prices_seen[-50:]