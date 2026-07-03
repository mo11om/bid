"""Tests for situation-triggered prompt blocks.

Detection must fire on the bench error families it targets (auction/hand
shapes from the §13 error analysis) and stay quiet on ordinary positions —
the confined-blast-radius property the mechanism exists for.
"""

import pytest

from src.config import Config
from src.harness.prompt_builder import ContextBuilder
from src.harness.situations import (
    ALL_TAGS,
    SITUATION_BLOCKS,
    detect_situations,
    parse_situations_setting,
)
from src.schema.dataset import MockDealRecord


# --------------------------------------------------------------------------- #
# double_candidate
# --------------------------------------------------------------------------- #
def test_double_fires_on_strong_hand_facing_their_contract():
    # bench-94 shape: 18 HCP, opponents drove to 5D.
    tags = detect_situations(
        "S:82.H:AQT62.D:AKJ.C:A82", 18,
        ["1D", "Pass", "1H", "1S", "2C", "Pass", "2S", "Pass", "3C", "Pass", "5D"],
    )
    assert "double_candidate" in tags


def test_double_fires_in_balancing_seat():
    # bench-55 shape: their 3C followed by two passes; pass-out seat.
    tags = detect_situations(
        "S:42.H:KQJ3.D:JT3.C:K863", 10,
        ["Pass", "Pass", "Pass", "1D", "2C", "2D", "3C", "Pass", "Pass"],
    )
    assert "double_candidate" in tags


def test_double_fires_on_their_suit_stacked():
    # bench-98 shape: KQ65 of hearts over RHO's 1H (responsive X spot).
    tags = detect_situations("S:.H:KQ65.D:642.C:KT9532", 8, ["1C", "X", "1H"])
    assert "double_candidate" in tags


def test_double_quiet_when_partner_made_last_bid():
    assert "double_candidate" not in detect_situations(
        "S:AK52.H:Q74.D:K6.C:9853", 13, ["Pass", "1NT", "Pass"]
    )


def test_double_quiet_on_weak_hand_no_stack_not_balancing():
    # bench-4 shape: 10 HCP over RHO's 1NT — existing example handles this.
    assert "double_candidate" not in detect_situations(
        "S:Q52.H:6543.D:K732.C:AJ", 10, ["Pass", "Pass", "1S", "1NT"]
    )


# --------------------------------------------------------------------------- #
# partner_game_drive
# --------------------------------------------------------------------------- #
def test_drive_fires_after_partner_takeout_x():
    # bench-28: LHO opened 1D, partner made a takeout X; we hold 16 with 6-5.
    tags = detect_situations(
        "S:54.H:.D:AQ953.C:AKQJT6", 16, ["Pass", "1D", "X", "1S"]
    )
    assert "partner_game_drive" in tags


def test_drive_fires_on_fit_for_partners_free_bid():
    # bench-33: we opened 1D, LHO X'd, partner bid 1S — we hold KJ93 support.
    tags = detect_situations(
        "S:KJ93.H:KQ64.D:T74.C:73", 9,
        ["Pass", "1D", "X", "1S", "Pass", "2C", "Pass", "4D", "Pass"],
    )
    assert "partner_game_drive" in tags


def test_drive_fires_on_partner_splinter():
    # bench-61 shape: we opened 1S, partner jumped 4C.
    tags = detect_situations(
        "S:QJT.H:KT73.D:T86.C:K92", 9, ["Pass", "1S", "Pass", "4C", "Pass"]
    )
    assert "partner_game_drive" in tags


def test_drive_fires_for_the_splinterer_continuing():
    # bench-63 shape: partner opened 1S, WE splintered 4C, partner cue'd 4D.
    tags = detect_situations(
        "S:AK74.H:A9.D:QJ4.C:QJ75", 17,
        ["Pass", "1S", "Pass", "4C", "Pass", "4D", "Pass"],
    )
    assert "partner_game_drive" in tags


def test_drive_quiet_on_plain_response():
    assert "partner_game_drive" not in detect_situations(
        "S:KQ72.H:8.D:JT64.C:A953", 10, ["1H", "Pass"]
    )


# --------------------------------------------------------------------------- #
# nt_game_candidate
# --------------------------------------------------------------------------- #
def test_nt_fires_with_values_and_stopper():
    # bench-41 shape: partner opened 1D, RHO overcalled 1H, we hold 14 w/ KT3.
    tags = detect_situations("S:J32.H:KT3.D:AKJT.C:Q65", 14, ["1D", "1H"])
    assert "nt_game_candidate" in tags


def test_nt_fires_over_preempt_with_stopper():
    # bench-78 shape: partner 1NT, RHO preempted 3D; KQJ83 behind them.
    tags = detect_situations("S:.H:AT64.D:KQJ83.C:QT42", 12, ["1NT", "3D"])
    assert "nt_game_candidate" in tags


def test_nt_quiet_without_stopper():
    # Same auction, diamond stopper removed.
    assert "nt_game_candidate" not in detect_situations(
        "S:KQJ83.H:AT64.D:864.C:Q2", 12, ["1NT", "3D"]
    )


def test_nt_quiet_when_opponents_silent():
    assert "nt_game_candidate" not in detect_situations(
        "S:J32.H:KT3.D:AKJT.C:Q65", 14, ["1D", "Pass"]
    )


# --------------------------------------------------------------------------- #
# strong_2c
# --------------------------------------------------------------------------- #
def test_2c_fires_for_responder_and_opener():
    hand, hcp = "S:Q.H:J75.D:AQ8754.C:AK5", 16
    assert "strong_2c" in detect_situations(hand, hcp, ["2C", "Pass"])
    assert "strong_2c" in detect_situations(
        hand, hcp, ["2C", "Pass", "2D", "Pass", "2H", "Pass"]
    )


def test_2c_quiet_when_opponents_opened_2c():
    # 2C opened by RHO (1 call back) is their auction, not ours.
    assert "strong_2c" not in detect_situations(
        "S:AK52.H:Q74.D:K6.C:9853", 13, ["2C"]
    )


# --------------------------------------------------------------------------- #
# Generic quiet cases + wiring
# --------------------------------------------------------------------------- #
def test_nothing_fires_on_openings():
    assert detect_situations("S:AK52.H:Q74.D:K6.C:9853", 13, []) == []
    assert detect_situations(
        "S:AK52.H:Q74.D:K6.C:9853", 13, ["Pass", "Pass"]
    ) == []


def _record(hand, hcp, bidding):
    return MockDealRecord(
        id="s-1", seat="N", hand=hand, hcp=hcp, shape="4-3-2-4", ltc=7,
        current_bidding=bidding, expert_bid="Pass", deal_pbn="", all_hands={},
        dealer="N", vulnerability="None",
    )


def test_block_injected_only_when_fired():
    rec_quiet = _record("S:AK52.H:Q74.D:K6.C:9853", 13, [])
    rec_fired = _record("S:J32.H:KT3.D:AKJT.C:Q65", 14, ["1D", "1H"])
    b = ContextBuilder(prompt_style="examples", situations="all")
    assert "Situation notes" not in b.build_prompt(rec_quiet)
    fired_prompt = b.build_prompt(rec_fired)
    assert "Situation notes" in fired_prompt
    assert "GAME CHECK (notrump)" in fired_prompt


def test_situations_none_renders_legacy_prompt():
    rec = _record("S:J32.H:KT3.D:AKJT.C:Q65", 14, ["1D", "1H"])
    off = ContextBuilder(prompt_style="examples", situations="none").build_prompt(rec)
    assert "Situation notes" not in off


def test_situations_subset_only_injects_selected():
    rec = _record("S:82.H:AQT62.D:AKJ.C:A82", 18,
                  ["1D", "Pass", "1H", "1S", "2C"])
    # Both double_candidate (18 HCP... last bid 2C by RHO) and nt_game fire;
    # enabling only strong_2c injects nothing here.
    only_2c = ContextBuilder(prompt_style="examples", situations="strong_2c")
    assert "Situation notes" not in only_2c.build_prompt(rec)


def test_config_validates_situations():
    Config(situations="none")
    Config(situations="double_candidate,strong_2c")
    with pytest.raises(ValueError):
        Config(situations="bogus_tag")
    with pytest.raises(ValueError):
        parse_situations_setting("double_candidate,bogus")


def test_all_tags_have_blocks():
    assert set(SITUATION_BLOCKS) == set(ALL_TAGS)
