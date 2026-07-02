"""Deterministic hand facts for prompt grounding (Phase 2, Strategy 1).

Pure evaluation of a single hand into the derived *decision-facts* a bidder
reasons about — total points, Rule of 20, longest/biddable suits, balance,
stoppers, quick tricks and controls. The prompt layer injects these as an
informational reference so the model doesn't have to (mis)count them itself.

Everything here derives ONLY from the active seat's hand plus its HCP, so it
carries no masking risk: no other seat's cards are ever consulted. Depends only
on the dependency-free primitives in :mod:`src.bridge`, keeping it trivially
unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.bridge import (
    SUITS,
    classify_shape,
    compute_shape,
    parse_hand,
    suit_lengths,
)

SUIT_NAMES = {"S": "Spades", "H": "Hearts", "D": "Diamonds", "C": "Clubs"}
MAJORS = ("S", "H")


@dataclass
class HandFacts:
    """Derived, deterministic facts about a single hand."""

    hcp: int
    length_points: int
    total_points: int
    rule_of_20: int
    opening_values: bool

    longest_suits: List[str]      # suit letters at max length
    five_card_majors: List[str]   # major suit letters with length >= 5
    biddable_suits: List[str]     # suit letters with length >= 4

    balanced: bool
    stoppers: Dict[str, bool]     # suit letter -> stopped?

    quick_tricks: float
    controls: int


def _suit_stopped(ranks: str) -> bool:
    """A suit is stopped with the Ace, a guarded King (Kx), or Qxx."""
    length = len(ranks)
    if "A" in ranks:
        return True
    if "K" in ranks and length >= 2:
        return True
    if "Q" in ranks and length >= 3:
        return True
    return False


def _suit_quick_tricks(ranks: str) -> float:
    """Standard quick-trick count for one suit (best applicable holding)."""
    length = len(ranks)
    has_a, has_k, has_q = ("A" in ranks), ("K" in ranks), ("Q" in ranks)
    if has_a and has_k:
        return 2.0
    if has_a and has_q:
        return 1.5
    if has_a:
        return 1.0
    if has_k and has_q:
        return 1.0
    if has_k and length >= 2:
        return 0.5
    return 0.0


def compute_hand_facts(hand: str, hcp: int) -> HandFacts:
    """Evaluate ``hand`` into a :class:`HandFacts` bundle.

    ``hcp`` is passed in (already computed upstream in the masked view) rather
    than recomputed, so the reference block and the prompt header can never
    disagree.
    """
    holdings = parse_hand(hand)
    lengths = suit_lengths(holdings)

    length_points = sum(max(0, lengths[s] - 4) for s in SUITS)
    total_points = hcp + length_points

    two_longest = sorted(lengths.values(), reverse=True)[:2]
    rule_of_20 = hcp + sum(two_longest)
    opening_values = hcp >= 13 or rule_of_20 >= 20

    max_len = max(lengths.values())
    longest_suits = [s for s in SUITS if lengths[s] == max_len]
    five_card_majors = [s for s in MAJORS if lengths[s] >= 5]
    biddable_suits = [s for s in SUITS if lengths[s] >= 4]

    balanced = classify_shape(compute_shape(holdings)) == "Balanced"
    stoppers = {s: _suit_stopped(holdings[s]) for s in SUITS}

    quick_tricks = sum(_suit_quick_tricks(holdings[s]) for s in SUITS)
    controls = sum(
        2 * holdings[s].count("A") + holdings[s].count("K") for s in SUITS
    )

    return HandFacts(
        hcp=hcp,
        length_points=length_points,
        total_points=total_points,
        rule_of_20=rule_of_20,
        opening_values=opening_values,
        longest_suits=longest_suits,
        five_card_majors=five_card_majors,
        biddable_suits=biddable_suits,
        balanced=balanced,
        stoppers=stoppers,
        quick_tricks=quick_tricks,
        controls=controls,
    )
