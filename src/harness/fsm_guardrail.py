"""Bidding legality guardrail (Phase 2).

A lightweight finite-state check that a proposed call is *structurally* legal
given the auction so far. It does not judge bridge quality — only legality:

* ``Pass`` is always legal.
* A contract bid must outrank the last contract bid.
* ``X`` (double) is legal only over an opponent's undoubled contract bid.
* ``XX`` (redouble) is legal only over an opponent's double of our side's bid.
"""

from __future__ import annotations

from typing import List

from src.bridge import bid_rank, is_contract_bid, normalize_call


class BiddingFSM:
    """Validates the structural legality of calls within an auction."""

    def is_valid_bid(self, history: List[str], new_bid: str) -> bool:
        """Return True if ``new_bid`` is legal following ``history``.

        ``history`` is the list of calls so far (dealer first). Seats are
        inferred positionally: index ``i`` is seat ``i % 4``; sides alternate.
        """
        try:
            call = normalize_call(new_bid)
        except ValueError:
            return False

        norm_history: List[str] = []
        for h in history:
            try:
                norm_history.append(normalize_call(h))
            except ValueError:
                return False  # corrupt history -> reject

        if call == "Pass":
            return True

        if call in ("X", "XX"):
            return self._double_legal(norm_history, call)

        # Contract bid: must strictly outrank the last contract bid (if any).
        last_rank = -1
        for h in norm_history:
            if is_contract_bid(h):
                last_rank = bid_rank(h)
        return bid_rank(call) > last_rank

    # ------------------------------------------------------------------ #
    def _double_legal(self, history: List[str], call: str) -> bool:
        """Legality of X / XX given the auction state.

        We scan backwards past trailing passes to the last meaningful call.
        """
        # Index of the seat about to call.
        seat = len(history) % 4

        last_meaningful = None
        last_meaningful_idx = None
        for i in range(len(history) - 1, -1, -1):
            if history[i] != "Pass":
                last_meaningful = history[i]
                last_meaningful_idx = i
                break

        if last_meaningful is None:
            return False  # nothing to double/redouble

        last_seat = last_meaningful_idx % 4
        same_side = (seat % 2) == (last_seat % 2)

        if call == "X":
            # Double an opponent's contract bid that isn't already doubled.
            return is_contract_bid(last_meaningful) and not same_side
        # XX: redouble an opponent's double of our side's contract.
        return last_meaningful == "X" and not same_side
