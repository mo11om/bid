"""Unit tests for deterministic hand-fact computation (pure, offline)."""

from src.bridge import count_hcp
from src.harness.hand_facts import compute_hand_facts


def _facts(hand: str):
    return compute_hand_facts(hand, count_hcp(hand))


def test_balanced_4333():
    # 4-3-3-3, 13 HCP: balanced, no length points, single longest suit.
    f = _facts("S:AK7.H:QJ3.D:854.C:KT62")
    assert f.balanced is True
    assert f.length_points == 0
    assert f.total_points == f.hcp == 13
    assert f.longest_suits == ["C"]           # the only 4-card suit
    assert f.five_card_majors == []
    assert f.biddable_suits == ["C"]
    assert f.opening_values is True           # 13 HCP


def test_five_card_major_flagged():
    # Spades AKQ32 (5), balanced 5-3-3-2, a stopped, opening values.
    f = _facts("S:AKQ32.H:K43.D:Q43.C:54")
    assert f.five_card_majors == ["S"]
    assert f.longest_suits == ["S"]
    assert f.length_points == 1               # one card past the 4th in spades
    assert f.balanced is True                 # 5-3-3-2


def test_void_and_singleton_stoppers():
    # 6-6-1-0: void in clubs, singleton King in diamonds (both unstopped).
    f = _facts("S:AKQJ54.H:AKQJ54.D:K.C:")  # D:K singleton, C void
    assert f.stoppers["C"] is False           # void
    assert f.stoppers["D"] is False           # bare King, length 1
    assert f.stoppers["S"] is True            # has Ace
    assert f.stoppers["H"] is True            # has Ace
    assert f.length_points == 4               # two 6-card suits, +2 each


def test_rule_of_20_boundary():
    # 11 HCP with 5-4 in the two longest suits -> Rule of 20 exactly 20 -> opens,
    # even though HCP alone (11) is below the 13 threshold.
    f = _facts("S:AKJ32.H:K876.D:43.C:32")
    assert f.hcp == 11
    assert f.rule_of_20 == 20                  # 11 + 5 + 4
    assert f.opening_values is True


def test_below_opening_values():
    # 10 HCP, flat 4-3-3-3 -> Rule of 20 = 17, no opening values.
    f = _facts("S:KQ72.H:K43.D:Q43.C:432")
    assert f.opening_values is False
    assert f.rule_of_20 == 17                  # 10 + 4 + 3


def test_quick_tricks_combos():
    # AK=2, AQ=1.5, lone K guarded (Kx)=0.5, bare K (length 1)=0.
    assert _facts("S:AK32.H:432.D:432.C:432").quick_tricks == 2.0
    assert _facts("S:AQ32.H:432.D:432.C:432").quick_tricks == 1.5
    # Kx in spades (0.5) + nothing else.
    assert _facts("S:K2.H:5432.D:5432.C:32").quick_tricks == 0.5
    # Singleton King is not a quick trick.
    assert _facts("S:K.H:5432.D:5432.C:432").quick_tricks == 0.0


def test_controls_count():
    # Two aces (4) + two kings (2) = 6 controls.
    f = _facts("S:AK32.H:AK32.D:432.C:32")
    assert f.controls == 6


def test_four_four_longest_tie():
    # 4-4-3-2: both 4-card suits are longest and both are biddable.
    f = _facts("S:AK32.H:QJ54.D:876.C:98")
    assert f.longest_suits == ["S", "H"]
    assert f.biddable_suits == ["S", "H"]
    assert f.balanced is True                  # 4-4-3-2
