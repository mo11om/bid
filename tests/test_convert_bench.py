"""Tests for the benchmark converter's deal reconstruction."""

import pytest

from scripts_convert_bench import group_deals, reconstruct_hands

# A real 52-card deal (the first deal of the Ben-SAYC set).
_HANDS = [
    "S:Q52 H:6543 D:K732 C:AJ",
    "S:97 H:A9872 D:AJ65 C:43",
    "S:J83 H:QJT D:QT C:K9852",
    "S:AKT64 H:K D:984 C:QT76",
]


def _rows(hands, auctions):
    return [{"hand": h, "auction": a} for h, a in zip(hands, auctions)]


def test_group_deals_splits_on_empty_auction():
    rows = _rows(
        _HANDS + [_HANDS[0]],
        ["", "Pass", "Pass Pass", "Pass Pass 1S", ""],
    )
    deals = group_deals(rows)
    assert [len(d) for d in deals] == [4, 1]


def test_reconstruct_valid_deal():
    rows = _rows(_HANDS, ["", "Pass", "Pass Pass", "Pass Pass 1S"])
    labeled = reconstruct_hands(rows, 0)
    assert set(labeled) == {"N", "E", "S", "W"}
    assert labeled["W"] == "S:AKT64.H:K.D:984.C:QT76"


def test_reconstruct_short_deal_returns_none():
    rows = _rows(_HANDS[:2], ["", "Pass"])
    assert reconstruct_hands(rows, 0) is None


def test_reconstruct_rejects_duplicate_cards():
    # Same hand four times -> only 13 unique cards.
    rows = _rows([_HANDS[0]] * 4, ["", "Pass", "Pass Pass", "Pass Pass 1S"])
    with pytest.raises(SystemExit):
        reconstruct_hands(rows, 0)
