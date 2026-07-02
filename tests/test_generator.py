"""Tests for record construction (decoupled from redeal)."""

from src.data.mock_generator import (
    dealer_for_board,
    records_from_labeled,
    vulnerability_for_board,
)

LABELED = {
    "N": "S:AK7.H:QJ3.D:854.C:KT62",   # 13 HCP -> opens 1NT
    "E": "S:432.H:432.D:432.C:5432",   # 0 HCP
    "S": "S:T98.H:T98.D:T98.C:T987",   # 0 HCP
    "W": "S:765.H:765.D:765.C:6543",   # 0 HCP
}


def test_board_metadata():
    assert dealer_for_board(1) == "N"
    assert dealer_for_board(2) == "E"
    assert vulnerability_for_board(1) == "None"
    assert vulnerability_for_board(2) == "NS"


def test_records_from_labeled_structure():
    recs = records_from_labeled(LABELED, board_number=1)
    assert len(recs) == 4
    assert [r.seat for r in recs] == ["N", "E", "S", "W"]  # from dealer N
    assert all(r.dealer == "N" and r.vulnerability == "None" for r in recs)
    # Full-information fields present on every record.
    for r in recs:
        assert set(r.all_hands) == {"N", "E", "S", "W"}
        assert r.deal_pbn.startswith("N:")


def test_records_auction_progression_and_expert():
    recs = records_from_labeled(LABELED, board_number=1)
    by_seat = {r.seat: r for r in recs}
    assert by_seat["N"].expert_bid == "1NT"   # 13 HCP, first to act
    assert by_seat["E"].expert_bid == "Pass"  # contract already on table
    assert by_seat["N"].current_bidding == []
    assert by_seat["E"].current_bidding == ["1NT"]
    assert by_seat["S"].current_bidding == ["1NT", "Pass"]
    assert by_seat["W"].current_bidding == ["1NT", "Pass", "Pass"]


def test_hand_evaluations_attached():
    recs = records_from_labeled(LABELED, board_number=1)
    north = next(r for r in recs if r.seat == "N")
    assert north.hcp == 13
    assert north.shape == "4-3-3-3"
    assert north.ltc == 8
