"""Unit tests for pure bridge primitives (offline, no redeal/endplay)."""

from src.bridge import (
    auction_is_closed,
    bid_rank,
    classify_shape,
    compute_ltc,
    compute_shape,
    count_hcp,
    hcp_bucket,
    normalize_call,
    parse_hand,
    seat_to_act,
    to_pbn_hand,
)

HAND = "S:AK7.H:QJ3.D:854.C:KT62"  # known: 4+2+0+3 = 9 HCP, 4-4-3-2 -> 4-3-3-3? check


def test_parse_hand_labeled_and_plain():
    labeled = parse_hand("S:AK7.H:QJ3.D:854.C:KT62")
    plain = parse_hand("AK7.QJ3.854.KT62")
    assert labeled == plain
    assert labeled["S"] == "AK7"
    assert labeled["C"] == "KT62"


def test_parse_hand_void():
    h = parse_hand("S:.H:AKQJ.D:AKQJ.C:AKQJT")
    assert h["S"] == ""
    assert len(h["C"]) == 5


def test_count_hcp():
    # AK=7, QJ=3, (854)=0, K=3 -> 13
    assert count_hcp(HAND) == 13
    # AKQJ(10) + AKQ(9) + AKQ(9) + 432(0) = 28, 13 cards.
    assert count_hcp("S:AKQJ.H:AKQ.D:AKQ.C:432") == 28


def test_compute_shape():
    # lengths S3 H3 D3 C4 -> sorted desc 4-3-3-3
    assert compute_shape(HAND) == "4-3-3-3"
    # S5 H3 D3 C2 -> 5-3-3-2
    assert compute_shape("S:AKQJT.H:AKQ.D:AKQ.C:AK") == "5-3-3-2"


def test_classify_shape():
    assert classify_shape("4-3-3-3") == "Balanced"
    assert classify_shape("4-4-3-2") == "Balanced"
    assert classify_shape("5-3-3-2") == "Balanced"
    assert classify_shape("5-5-2-1") == "Two-suited"
    assert classify_shape("6-4-2-1") == "Two-suited"
    assert classify_shape("5-4-3-1") == "Two-suited"
    assert classify_shape("6-3-2-2") == "Unbalanced"  # single-suiter
    assert classify_shape("7-3-2-1") == "Unbalanced"
    assert classify_shape("4-4-4-1") == "Unbalanced"  # three-suiter


def test_compute_ltc_known():
    # AK7: missing Q -> 1 loser. QJ3: missing A,K -> 2. 854: A,K,Q missing -> 3.
    # KT62: missing A,Q -> 2. Total = 1+2+3+2 = 8.
    assert compute_ltc(HAND) == 8


def test_compute_ltc_edges():
    assert compute_ltc("S:.H:.D:.C:AKQJT98765432") == 0  # void,void,void, AKQ held
    # Singleton ace = 0 losers; singleton small = 1 loser.
    assert compute_ltc("S:A.H:2.D:AKQ.C:AKQJT987") == 1  # only the H singleton small loses


def test_normalize_call():
    assert normalize_call("pass") == "Pass"
    assert normalize_call("p") == "Pass"
    assert normalize_call("1n") == "1NT"
    assert normalize_call("3H") == "3H"
    assert normalize_call("dbl") == "X"
    assert normalize_call("redouble") == "XX"


def test_bid_rank_ordering():
    assert bid_rank("1C") < bid_rank("1NT") < bid_rank("2C")
    assert bid_rank("Pass") == -1
    assert bid_rank("7NT") == 34


def test_seat_to_act():
    assert seat_to_act("N", 0) == "N"
    assert seat_to_act("N", 1) == "E"
    assert seat_to_act("E", 2) == "W"


def test_auction_is_closed():
    assert not auction_is_closed([])
    assert auction_is_closed(["Pass", "Pass", "Pass", "Pass"])  # passed out
    assert auction_is_closed(["1NT", "Pass", "Pass", "Pass"])
    assert not auction_is_closed(["1NT", "Pass", "Pass"])
    assert not auction_is_closed(["Pass", "1H", "Pass", "Pass"])
    assert auction_is_closed(["Pass", "1H", "Pass", "Pass", "Pass"])


def test_to_pbn_hand():
    assert to_pbn_hand(HAND) == "AK7.QJ3.854.KT62"


def test_hcp_bucket():
    assert hcp_bucket(0) == "0-10"
    assert hcp_bucket(10) == "0-10"
    assert hcp_bucket(11) == "11-15"
    assert hcp_bucket(16) == "16+"
