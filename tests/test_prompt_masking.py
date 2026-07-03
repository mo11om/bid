"""Prompt tests: masking, feature injection, and auction-role annotation."""

from src.harness.prompt_builder import (
    EXAMPLES_BLOCK,
    KEPT_RULES,
    SAYC_KNOWLEDGE,
    ContextBuilder,
    annotate_auction,
)
from src.schema.dataset import MockDealRecord

builder = ContextBuilder()


def _dynamic(prompt: str) -> str:
    """Strip the static knowledge/rules/examples text, leaving only the
    record-derived content — the part the masking guarantee is about. The
    static few-shot examples legitimately contain card strings."""
    for block in (SAYC_KNOWLEDGE, KEPT_RULES, EXAMPLES_BLOCK):
        prompt = prompt.replace(block, "")
    return prompt


def _record(current_bidding=None) -> MockDealRecord:
    return MockDealRecord(
        id="t1",
        seat="N",
        hand="S:AK7.H:QJ3.D:854.C:KT62",
        hcp=13,
        shape="4-3-3-3",
        ltc=8,
        current_bidding=current_bidding if current_bidding is not None else ["Pass"],
        expert_bid="1NT",
        deal_pbn="N:AK7.QJ3.854.KT62 E:QJ2.A8.AKQ.AQJ9 S:T98.KT9.JT9.8753 W:6543.7654.762.4",
        all_hands={
            "N": "AK7.QJ3.854.KT62",
            "E": "QJ2.A8.AKQ.AQJ9",
            "S": "T98.KT9.JT9.8753",
            "W": "6543.7654.762.4",
        },
        dealer="N",
        vulnerability="None",
    )


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def test_prompt_contains_active_hand_only():
    rec = _record()
    prompt = builder.build_prompt(rec)
    # Active hand is rendered space-separated per suit, e.g. "A K 7".
    assert "A K 7" in prompt
    assert "13 HCP" in prompt
    # No other seat's distinctive holdings leak into the record-derived part
    # of the prompt (checked both contiguous and spaced, since only the active
    # hand is ever rendered).
    dynamic = _dynamic(prompt)
    for leak in ("AQJ9", "A Q J 9", "8753", "8 7 5 3", "6543", "6 5 4 3"):
        assert leak not in dynamic


def test_build_prompt_parts_matches():
    rec = _record()
    p1 = builder.build_prompt(rec)
    p2 = builder.build_prompt_parts("N", rec.hand, rec.hcp, list(rec.current_bidding))
    assert p1 == p2


# --------------------------------------------------------------------------- #
# Feature injection
# --------------------------------------------------------------------------- #
def test_prompt_injects_suit_lengths_and_shape():
    rec = _record()
    prompt = builder.build_prompt(rec)
    assert "4 cards" in prompt  # clubs KT62
    assert "3 cards" in prompt
    assert "Distribution: 4-3-3-3" in prompt
    assert "Balanced" in prompt
    assert "LTC" in prompt


def test_prompt_classifies_two_suited_hand():
    prompt = builder.build_prompt_parts(
        "N", "S:AKQ32.H:KQJ54.D:8.C:74", 13, []
    )
    assert "Distribution: 5-5-2-1" in prompt
    assert "Two-suited" in prompt


# --------------------------------------------------------------------------- #
# Reference (computed-facts) block
# --------------------------------------------------------------------------- #
def test_prompt_injects_reference_facts_block():
    rec = _record()  # 4-3-3-3, 13 HCP
    prompt = builder.build_prompt(rec)
    assert "Reference values (computed from your hand):" in prompt
    assert "opening values: yes" in prompt        # 13 HCP -> opens
    assert "Balanced: yes" in prompt
    assert "Quick tricks:" in prompt and "Controls:" in prompt


def test_reference_block_never_leaks_other_hands():
    # The facts block is single-hand only; no other seat's holdings appear.
    rec = _record()
    dynamic = _dynamic(builder.build_prompt(rec))
    for leak in ("AQJ9", "A Q J 9", "8753", "8 7 5 3", "6543", "6 5 4 3"):
        assert leak not in dynamic


# --------------------------------------------------------------------------- #
# Auction-role annotation
# --------------------------------------------------------------------------- #
def test_annotate_auction_opening():
    annotated, summary = annotate_auction([])
    assert "first to call" in annotated
    assert "OPENING" in summary


def test_annotate_auction_all_pass_is_opening():
    _, summary = annotate_auction(["Pass", "Pass", "Pass"])
    assert "OPENING" in summary or "balancing" in summary


def test_annotate_auction_partner_opened():
    # history from active seat's view: LHO, Partner, RHO → partner opened 1NT.
    annotated, summary = annotate_auction(["Pass", "1NT", "Pass"])
    assert "1NT(Partner)" in annotated
    assert "RESPONDING" in summary
    assert "partner opened 1NT" in summary


def test_annotate_auction_opponent_opened():
    # Single prior call is the RHO's opening.
    annotated, summary = annotate_auction(["1H"])
    assert "1H(RHO)" in annotated
    assert "opponent opened 1H" in summary
    assert "competing" in summary.lower() or "defending" in summary.lower()


def test_role_summary_appears_in_prompt():
    rec = _record(current_bidding=["Pass", "1NT", "Pass"])
    prompt = builder.build_prompt(rec)
    assert "RESPONDING" in prompt
    assert "1NT(Partner)" in prompt
