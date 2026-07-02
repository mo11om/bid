"""Masked prompt assembly (Phase 2).

The prompt contains ONLY the active seat's hand, its HCP, derived features of
that hand (suit lengths, distribution, LTC), and the auction so far — annotated
with table roles (partner vs opponent) inferred purely from auction *order*.
Other hands are never referenced. ``build_prompt`` reads exclusively from
``MockDealRecord.masked_view()`` to make the masking guarantee structural.
"""

from __future__ import annotations

from typing import List, Tuple

from src.bridge import (
    SUITS,
    classify_shape,
    compute_ltc,
    compute_shape,
    parse_hand,
)
from src.harness.hand_facts import HandFacts, SUIT_NAMES, compute_hand_facts
from src.schema.dataset import MockDealRecord

_SUIT_SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}

# Roles of the three prior seats, back-counted from the active seat: the call
# immediately before the active seat is the right-hand opponent, then partner,
# then the left-hand opponent (then the active seat itself, repeating).
_ROLE_CYCLE = ("RHO", "Partner", "LHO")

SYSTEM_INSTRUCTION = (
    "You are an expert contract bridge player bidding under Standard American "
    "(SAYC). Given your hand and the auction so far, choose your single best "
    "call. Respond ONLY with a JSON object of the form "
    '{"thinking": "<short reasoning>", "bid": "<call>"} where <call> is one of '
    "Pass, X, XX, or a contract bid like 1NT, 3H, 4S, 6C. "
    "Output exactly one JSON object and nothing else — no markdown fences, no "
    "text before or after it."
)


def annotate_auction(history: List[str]) -> Tuple[str, str]:
    """Annotate the auction with table roles and summarize the active seat's task.

    Roles are derived only from position in the auction relative to the active
    seat (no dealer, no hidden information): reading backward from the end, the
    calls are RHO, Partner, LHO, RHO, Partner, LHO, ... The returned summary
    tells the model whether it is opening, responding to partner, or competing.
    """
    if not history:
        return "(you are first to call)", "You are OPENING the auction."

    n = len(history)
    roles = [_ROLE_CYCLE[(n - 1 - i) % 3] for i in range(n)]
    annotated = " ".join(f"{call}({role})" for call, role in zip(history, roles))

    # Find the first non-Pass call — that is the opening bid.
    opening_idx = next((i for i, c in enumerate(history) if c != "Pass"), None)
    if opening_idx is None:
        summary = "No one has opened yet. You are OPENING (or balancing)."
    else:
        opener_role = roles[opening_idx]
        opening_bid = history[opening_idx]
        if opener_role == "Partner":
            summary = (
                f"Your partner opened {opening_bid}. You are RESPONDING, "
                f"not opening — bid in support of or in reply to partner."
            )
        else:
            side = "left-hand" if opener_role == "LHO" else "right-hand"
            summary = (
                f"Your {side} opponent opened {opening_bid}. You are competing/"
                f"defending, not opening."
            )
    return annotated, summary


class ContextBuilder:
    """Builds masked bidding prompts for a single seat."""

    def build_prompt(self, record: MockDealRecord) -> str:
        view = record.masked_view()
        return self._render(
            seat=str(view["seat"]),
            hand=str(view["hand"]),
            hcp=int(view["hcp"]),
            history=list(view["current_bidding"]),
        )

    def build_prompt_parts(
        self, seat: str, hand: str, hcp: int, history: List[str]
    ) -> str:
        """Same prompt from raw parts (used by the self-play rollout)."""
        return self._render(seat=seat, hand=hand, hcp=hcp, history=history)

    # ------------------------------------------------------------------ #
    def _render(self, seat: str, hand: str, hcp: int, history: List[str]) -> str:
        holdings = parse_hand(hand)
        hand_lines = "\n".join(
            f"  {_SUIT_SYMBOL[s]} {len(holdings[s])} cards "
            f"({' '.join(holdings[s]) if holdings[s] else '—'})"
            for s in SUITS
        )
        shape = compute_shape(hand)
        summary_line = (
            f"Distribution: {shape}  ·  {classify_shape(shape)}  ·  "
            f"LTC {compute_ltc(hand)}"
        )
        facts_block = self._render_facts(compute_hand_facts(hand, hcp))
        annotated, role_summary = annotate_auction(history)
        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"Your seat: {seat}\n"
            f"Your hand ({hcp} HCP):\n{hand_lines}\n"
            f"{summary_line}\n\n"
            f"{facts_block}\n\n"
            f"Auction so far (dealer first): {annotated}\n"
            f"→ {role_summary}\n\n"
            f"Your call:"
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_facts(facts: HandFacts) -> str:
        """Format a :class:`HandFacts` bundle as an informational reference block.

        Neutral, non-imperative framing: the values are offered as a computed
        reference, not asserted as rules the model must obey.
        """
        def names(letters: list[str]) -> str:
            return ", ".join(SUIT_NAMES[s] for s in letters) if letters else "none"

        longest = ", ".join(SUIT_NAMES[s] for s in facts.longest_suits)
        stoppers = " ".join(
            f"{_SUIT_SYMBOL[s]} {'yes' if facts.stoppers[s] else 'no'}" for s in SUITS
        )
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        qt = f"{facts.quick_tricks:g}"
        return (
            "Reference values (computed from your hand):\n"
            f"  • HCP {facts.hcp} · length pts +{facts.length_points} · "
            f"total pts {facts.total_points}\n"
            f"  • Rule of 20: {facts.rule_of_20}  ·  "
            f"opening values: {yn(facts.opening_values)}\n"
            f"  • Longest suit(s): {longest}   · "
            f"5-card majors: {names(facts.five_card_majors)}\n"
            f"  • Biddable (4+) suits: {names(facts.biddable_suits)}\n"
            f"  • Balanced: {yn(facts.balanced)}   · Stoppers: {stoppers}\n"
            f"  • Quick tricks: {qt} · Controls: {facts.controls}"
        )
