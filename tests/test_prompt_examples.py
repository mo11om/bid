"""Tests for the few-shot examples block and prompt-style dispatch.

The most important test here is the anti-leakage fence: bridge-llm-bench's own
example block contained test-set deals, which inflated its headline accuracy.
Every example hand in this harness must stay out of the Ben-SAYC benchmark
data, forever.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.bridge import count_hcp, parse_hand, suit_lengths
from src.config import Config
from src.harness.prompt_builder import (
    EXAMPLES,
    EXAMPLES_BLOCK,
    KEPT_RULES,
    SAYC_KNOWLEDGE,
    ContextBuilder,
    annotate_auction,
)
from src.schema.dataset import MockDealRecord

ROOT = Path(__file__).resolve().parent.parent
BENCH25 = ROOT / "data" / "bench25_bensayc.jsonl"
BENCH_CSV = ROOT.parent / "bridge-llm-bench" / "data" / "ben_sayc_100.csv"


def _dotted(hand: str) -> str:
    """'S:Q52 H:6543 ...' -> 'S:Q52.H:6543....' (parse_hand needs dots)."""
    return ".".join(hand.split())


def _normalize(hand: str) -> str:
    """Canonical form for hand comparison across formats/orderings."""
    holdings = parse_hand(_dotted(hand))
    return "|".join(f"{s}:{holdings[s]}" for s in "SHDC")


def _benchmark_hands() -> set[str]:
    hands: set[str] = set()
    if BENCH25.exists():
        for line in open(BENCH25):
            if line.strip():
                hands.add(_normalize(json.loads(line)["hand"]))
    if BENCH_CSV.exists():
        for row in csv.DictReader(open(BENCH_CSV)):
            hands.add(_normalize(row["hand"]))
    return hands


RECORD = MockDealRecord(
    id="t-1",
    seat="N",
    hand="S:AK52.H:Q74.D:K6.C:9853",
    hcp=13,
    shape="4-3-2-4",
    ltc=7,
    current_bidding=[],
    expert_bid="1C",
    deal_pbn="",
    all_hands={},
    dealer="N",
    vulnerability="None",
)


# --------------------------------------------------------------------------- #
# Anti-leakage fence
# --------------------------------------------------------------------------- #
def test_no_example_hand_appears_in_benchmark_data():
    if not BENCH25.exists() and not BENCH_CSV.exists():
        pytest.skip(
            "benchmark data not present (fresh clone?) — regenerate with "
            "scripts_convert_bench.py to arm the leakage fence"
        )
    bench = _benchmark_hands()
    assert bench, "benchmark data present but empty — fence would be vacuous"
    for hand, _info, _auction, _call in EXAMPLES:
        assert _normalize(hand) not in bench, (
            f"example hand {hand!r} appears in the Ben-SAYC benchmark data — "
            f"this is test-set leakage; replace it with a fresh hand"
        )


# --------------------------------------------------------------------------- #
# Example self-consistency (each hand is legal and annotated correctly)
# --------------------------------------------------------------------------- #
def test_example_hands_have_13_cards():
    for hand, _info, _auction, _call in EXAMPLES:
        assert sum(suit_lengths(_dotted(hand)).values()) == 13, hand


def test_example_hcp_annotations_are_correct():
    for hand, info, _auction, _call in EXAMPLES:
        stated = int(info.split(" HCP")[0])
        actual = count_hcp(_dotted(hand))
        assert actual == stated, (
            f"{hand}: annotation says {stated} HCP, actual {actual}"
        )


# --------------------------------------------------------------------------- #
# Prompt-style dispatch
# --------------------------------------------------------------------------- #
def test_base_style_has_no_knowledge_or_examples():
    prompt = ContextBuilder(prompt_style="base").build_prompt(RECORD)
    assert "SAYC Complete Reference" not in prompt
    assert "Examples (study carefully)" not in prompt
    assert "PENALTY DOUBLES" not in prompt


def test_knowledge_style_has_knowledge_only():
    prompt = ContextBuilder(prompt_style="knowledge").build_prompt(RECORD)
    assert "SAYC Complete Reference" in prompt
    assert "Examples (study carefully)" not in prompt
    assert "PENALTY DOUBLES" not in prompt


def test_examples_style_has_all_blocks():
    prompt = ContextBuilder(prompt_style="examples").build_prompt(RECORD)
    assert SAYC_KNOWLEDGE in prompt
    assert KEPT_RULES in prompt
    assert EXAMPLES_BLOCK in prompt


def test_examples_is_the_default_style():
    assert ContextBuilder().prompt_style == "examples"
    assert Config().prompt_style == "examples"


def test_invalid_style_rejected():
    with pytest.raises(ValueError):
        ContextBuilder(prompt_style="fancy")
    with pytest.raises(ValueError):
        Config(prompt_style="fancy")


# --------------------------------------------------------------------------- #
# Auction role annotation: the cycle has period 4 (own calls included)
# --------------------------------------------------------------------------- #
def test_own_earlier_call_labeled_you():
    # N deals: N=Pass(You), E=Pass(LHO), S=1S(Partner), W=1NT(RHO); N to act.
    annotated, summary = annotate_auction(["Pass", "Pass", "1S", "1NT"])
    assert annotated == "Pass(You) Pass(LHO) 1S(Partner) 1NT(RHO)"
    assert "partner opened 1s" in summary.lower()


def test_partner_opening_seen_across_full_round():
    # 8 calls back from the active seat: partner opened 1S (bench-8 shape).
    history = ["Pass", "Pass", "1S", "1NT", "Pass", "2D", "Pass", "2H"]
    _, summary = annotate_auction(history)
    assert "partner opened 1s" in summary.lower()


def test_own_opening_recognized():
    # 5 calls: you opened 1H two rounds ago (bench-18 shape).
    history = ["Pass", "1H", "Pass", "2C", "X"]
    annotated, summary = annotate_auction(history)
    assert "1H(You)" in annotated
    assert "YOU opened 1H" in summary


def test_rho_opening_not_mislabeled_partner():
    # 6 calls: RHO opened 1H (bench-19 shape — was mislabeled "partner").
    history = ["Pass", "1H", "Pass", "2C", "X", "Pass"]
    annotated, summary = annotate_auction(history)
    assert "1H(RHO)" in annotated
    assert "X(Partner)" in annotated
    assert "right-hand opponent opened 1H" in summary
