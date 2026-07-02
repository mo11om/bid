"""Synthetic dataset generation (Phase 1).

Uses ``redeal`` to deal random boards, evaluates each hand with the shared
:mod:`src.bridge` utilities, and writes one :class:`MockDealRecord` per seat to
a JSONL file. The "expert" bid is a deliberately trivial placeholder heuristic
(HCP >= 12 in first-to-bid seat -> 1NT, else Pass) standing in for real
expert/BBO data later.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from src.bridge import (
    compute_ltc,
    compute_shape,
    count_hcp,
    is_contract_bid,
    to_labeled_hand,
    to_pbn_hand,
)
from src.schema.dataset import MockDealRecord

SEATS = ("N", "E", "S", "W")


# --------------------------------------------------------------------------- #
# redeal interop
# --------------------------------------------------------------------------- #
def _holding_str(holding) -> str:
    """Rank string (high->low) for a redeal Holding, e.g. 'AK7'."""
    # redeal Holding.__str__ already yields ranks high->low using 'T' for ten.
    return str(holding).replace("10", "T")


def _hand_to_labeled(hand) -> str:
    """Convert a redeal Hand to our labeled canonical form."""
    return (
        f"S:{_holding_str(hand.spades)}."
        f"H:{_holding_str(hand.hearts)}."
        f"D:{_holding_str(hand.diamonds)}."
        f"C:{_holding_str(hand.clubs)}"
    )


def _build_dealer():
    """Return a redeal dealer callable producing random deals.

    Imported lazily so the rest of the package (and its tests) work without
    redeal installed.
    """
    from redeal import Deal

    return Deal.prepare()


# --------------------------------------------------------------------------- #
# Board metadata
# --------------------------------------------------------------------------- #
def dealer_for_board(board_number: int) -> str:
    """Standard rotating dealer: board 1 -> N, 2 -> E, ..."""
    return SEATS[(board_number - 1) % 4]


def vulnerability_for_board(board_number: int) -> str:
    """Standard duplicate vulnerability schedule (16-board cycle)."""
    n = board_number - 1
    idx = (n + n // 4) % 4
    return ("None", "NS", "EW", "Both")[idx]


# --------------------------------------------------------------------------- #
# Auction heuristic (placeholder expert)
# --------------------------------------------------------------------------- #
def heuristic_call(hcp: int, auction: List[str]) -> str:
    """Trivial reference policy: open 1NT with 12+ HCP if nobody has bid yet."""
    has_contract = any(is_contract_bid(b) for b in auction)
    if hcp >= 12 and not has_contract:
        return "1NT"
    return "Pass"


# --------------------------------------------------------------------------- #
# Record construction
# --------------------------------------------------------------------------- #
def _deal_to_full_pbn(labeled_by_seat: Dict[str, str]) -> str:
    """Build the endplay PBN deal string 'N:.. E:.. S:.. W:..' (plain holdings)."""
    parts = [to_pbn_hand(labeled_by_seat[seat]) for seat in SEATS]
    return "N:" + " ".join(parts)


def records_for_deal(deal, board_number: int) -> List[MockDealRecord]:
    """Produce the four masked records for a single redeal Deal."""
    labeled = {
        "N": _hand_to_labeled(deal.north),
        "E": _hand_to_labeled(deal.east),
        "S": _hand_to_labeled(deal.south),
        "W": _hand_to_labeled(deal.west),
    }
    return records_from_labeled(labeled, board_number)


def records_from_labeled(
    labeled: Dict[str, str], board_number: int
) -> List[MockDealRecord]:
    """Build records from labeled hands (decoupled from redeal for testing)."""
    dealer = dealer_for_board(board_number)
    vul = vulnerability_for_board(board_number)
    all_hands_pbn = {seat: to_pbn_hand(labeled[seat]) for seat in SEATS}
    deal_pbn = _deal_to_full_pbn(labeled)

    # Walk the auction from the dealer, capturing each seat's turn.
    order = [SEATS[(SEATS.index(dealer) + i) % 4] for i in range(4)]
    auction: List[str] = []
    records: List[MockDealRecord] = []
    for seat in order:
        hand = labeled[seat]
        hcp = count_hcp(hand)
        expert = heuristic_call(hcp, auction)
        records.append(
            MockDealRecord(
                id=f"b{board_number}-{seat}",
                seat=seat,
                hand=to_labeled_hand(hand),
                hcp=hcp,
                shape=compute_shape(hand),
                ltc=compute_ltc(hand),
                current_bidding=list(auction),
                expert_bid=expert,
                deal_pbn=deal_pbn,
                all_hands=all_hands_pbn,
                dealer=dealer,
                vulnerability=vul,
            )
        )
        auction.append(expert)
    return records


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def generate_mock_dataset(count: int, output_path: str) -> int:
    """Generate ``count`` deals (4 records each) and write JSONL to ``output_path``.

    Returns the number of records written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dealer = _build_dealer()

    written = 0
    with open(output_path, "w", encoding="utf-8") as fh:
        for board in range(1, count + 1):
            deal = dealer()
            for record in records_for_deal(deal, board):
                fh.write(record.model_dump_json() + "\n")
                written += 1
    return written
