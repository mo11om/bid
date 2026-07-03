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

# Roles of prior calls, back-counted from the active seat: the call
# immediately before the active seat is the right-hand opponent, then partner,
# then the left-hand opponent, then the active seat's OWN earlier call — a
# bridge auction rotates four seats, so the cycle has period 4. (A period-3
# cycle here mislabels every call four or more back, including who opened.)
_ROLE_CYCLE = ("RHO", "Partner", "LHO", "You")

SYSTEM_INSTRUCTION = (
    "You are an expert contract bridge player bidding under Standard American "
    "(SAYC). Given your hand and the auction so far, choose your single best "
    "call. Respond ONLY with a JSON object of the form "
    '{"thinking": "<short reasoning>", "bid": "<call>"} where <call> is one of '
    "Pass, X, XX, or a contract bid like 1NT, 3H, 4S, 6C. "
    "Output exactly one JSON object and nothing else — no markdown fences, no "
    "text before or after it."
)

# SAYC reference guide ported verbatim from bridge-llm-bench
# (bridge_llm_bench/utils/config.py:SAYC_KNOWLEDGE). Injected by the
# "knowledge" and "examples" prompt styles (see Config.prompt_style).
SAYC_KNOWLEDGE = """\
SAYC Complete Reference:
OPENING BIDS: 12-21 HCP required. 5+ card major open 1H/1S (higher first with 5-5). \
No 5-card major: open longest minor (1C with 3-3, 1D with 4-4). \
15-17 balanced open 1NT (may have 5-card suit). 20-21 balanced open 2NT. \
22+ HCP open 2C (strong artificial, 2D=waiting). \
Weak 2 (2D/2H/2S) = 5-11 HCP + good 6-card suit, no void, no outside 4-card major. \
3-level preempt = 7-card suit too weak to open at 1. Pass with <12 HCP and no preempt shape.
RESPONSES TO 1H/1S: 6-9 raise with 3+ trump or bid 1NT (non-forcing); \
10-12 jump raise (limit raise, 3+ trump) or new suit; \
13+ Jacoby 2NT (4+ trump, game-forcing) or new suit forcing; \
jump to 4M with 5+ trump and <10 HCP (preemptive). New suit at 1-level=4+ cards, forcing.
RESPONSES TO 1C/1D: New suit at 1-level (4+ cards, bid up the line) preferred; \
raise minor with 5+ support; 1NT=6-10 no 4-card major; \
2NT=11-12 balanced no major; 3NT=13-15 balanced no major.
RESPONSES TO 1NT: 2C=Stayman (need 4-card major, 8+ HCP); \
2D=transfer to 2H, 2H=transfer to 2S (5+ cards); 2S=puppet to 3C; \
2NT=invitational; 4C=Gerber (ace-ask); 4NT=quantitative. 0-7 HCP Pass or transfer+pass.
RESPONSES TO 2C: 2D=artificial waiting; 2H/2S/3C/3D=natural GF 5+ cards with 2 of top 3; \
2NT=8+ balanced.
RESPONSES TO WEAK 2: 2NT=forcing (opener shows feature or rebids suit); \
raise=to play; new suit=5+ forcing one round.
OPENER REBIDS: Min(13-15) cheapest NT/raise/rebid suit; Med(16-18) jump raise/jump rebid/reverse; \
Max(19-21) jump NT/double-jump raise/jump shift. Reverse=new suit above 2 of opened suit, 16+ HCP.
COMPETITIVE: Overcall 1-level=8-16 HCP 5+ cards; 1NT overcall=15-18 balanced with stopper; \
Takeout X=support for unbid suits 12+ or 17+ any; X of game-level=penalty. \
Unusual 2NT=5-5+ two lowest unbid suits; Michaels cuebid=over minor 5-5 majors, over major 5-5 other major+minor. \
Jump overcall=preemptive. Negative X thru 3S=values in unbid suits.
AFTER OPP TAKEOUT X: Redouble=10+ tends to deny fit; 2NT Jordan=limit raise+; \
jump raise=preemptive; new suit at 1-level=forcing; jump shift=preemptive.
SLAM: Blackwood 4NT=ace ask (5C=0/4, 5D=1, 5H=2, 5S=3); Gerber 4C over NT=ace ask; \
Grand Slam Force 5NT=bid 7 with 2 of top 3 honors. Cue-bid controls once trump agreed.
PASSED HAND: may open lighter in 3rd/4th seat (10-11 HCP ok). \
Passed hand responses are non-forcing (no longer unlimited).\
"""

# Targeted rule blocks kept from bridge-llm-bench's ablation-tested P22 prompt.
# The ablation found most rule blocks neutral-to-harmful; only these survived.
# (Deliberately NOT ported: competitive-bidding rules, takeout-double-response
# rules, and "when not to compete" rules — each measurably hurt accuracy.)
KEPT_RULES = """\
PENALTY DOUBLES:
- When partner makes a PENALTY double (double of a suit at 2+ level), PASS to defend
- Do NOT pull partner's penalty double unless you have extreme distribution (void in their suit)

5-LEVEL DECISIONS:
- 'The 5-level belongs to the opponents' — do NOT compete to 5-minor unless forced
- Once the auction has reached the 5-level, STOP competing

OPENING SUIT CHOICE:
- With 6-5 in two suits, open the LONGER suit (6C+5D = open 1C, not 1D)
- With 5-5, open the HIGHER-ranking suit (5H+5S = open 1S)\
"""

# Few-shot examples in the format bridge-llm-bench's ablation proved essential
# (removing all examples cost -29pts; removing all rules cost -0.7pts). Every
# hand below is freshly composed — NONE may appear in the Ben-SAYC benchmark
# datasets. tests/test_prompt_examples.py enforces this: the benchmark's own
# example block leaked test-set deals, which inflated its headline number.
#
# Tuples: (hand, annotation, auction, call_and_reason). Auctions use the same
# call vocabulary as the live prompt ("Pass", "X", "1S", ...), dealer first.
EXAMPLES: List[Tuple[str, str, str, str]] = [
    # -- opening anchors -------------------------------------------------
    ("S:AQJ85 H:K72 D:96 C:K84", "13 HCP, 5S", "None",
     "1S (5-card major, opening values)"),
    ("S:KQ4 H:AJ6 D:KT52 C:Q97", "15 HCP, balanced", "None",
     "1NT (15-17 balanced)"),
    ("S:AK4 H:QT53 D:5 C:KJ742", "13 HCP, 5C", "None",
     "1C (no 5-card major — open the longest minor)"),
    # -- light-opening restraint ------------------------------------------
    ("S:K9752 H:QJ4 D:J83 C:Q6", "9 HCP, 5S", "None",
     "Pass (Rule of 20 fails: 9 HCP + 5 + 3 = 17 < 20. Do NOT open light)"),
    ("S:A87 H:KQ654 D:Q93 C:84", "11 HCP, 5H", "None",
     "Pass (Rule of 20: 11 + 5 + 3 = 19 < 20 — close, but Pass in 1st/2nd seat)"),
    ("S:83 H:A76542 D:K92 C:75", "7 HCP, 6H", "None",
     "Pass (A76542 has only one honor — suit too weak for a weak 2H)"),
    # -- simple responses --------------------------------------------------
    ("S:KQ72 H:8 D:JT64 C:A953", "10 HCP, 4S", "1H Pass",
     "1S (new suit at the 1-level is forcing — show the 4-card major)"),
    ("S:T853 H:K64 D:A72 C:865", "7 HCP, 3H", "1H Pass",
     "2H (simple raise: 6-9 points with 3-card support)"),
    # -- overcalls: when and when not ---------------------------------------
    ("S:AQJ96 H:84 D:K73 C:962", "10 HCP, 5S", "1D",
     "1S (1-level overcall: 8-16 HCP with a good 5-card suit)"),
    ("S:J4 H:964 D:KQT95 C:J82", "7 HCP, 5D", "1S",
     "Pass (too weak for a 2-level overcall — that needs ~10+ HCP and a good suit)"),
    # -- competitive auctions ----------------------------------------------
    ("S:K74 H:9542 D:A853 C:Q6", "9 HCP, 3S", "Pass Pass 1S 1NT",
     "X (partner opened 1S, RHO overcalled 1NT: competitive double with "
     "3-card support — do NOT sell out to 1NT)"),
    ("S:85 H:KT762 D:J943 C:82", "4 HCP, 5H", "Pass Pass 1S 1NT Pass",
     "2D (partner overcalled 1NT = 15-18; systems on — 2D transfers to hearts)"),
    ("S:KJ83 H:942 D:A76 C:T85", "8 HCP, 4S", "1S 2H",
     "3S (partner opened 1S, RHO overcalled: competitive jump raise with "
     "4 trumps — don't just Pass with a fit)"),
    ("S:Q93 H:AJ4 D:KQ85 C:J62", "13 HCP, balanced", "1D 1H",
     "3NT (partner opened, RHO overcalled: 13 HCP balanced + heart stopper "
     "AJ4 — bid the NT game)"),
    ("S:84 H:73 D:KQJ953 C:A62", "10 HCP, 6D", "Pass 1H Pass 2C X Pass",
     "3D (partner's X is takeout — JUMP to 3D with a strong 6-card suit, "
     "don't bid a timid 2D)"),
    ("S:A5 H:KQJ74 D:962 C:K83", "13 HCP, 5H", "Pass 1H Pass 2C X",
     "3C (you opened 1H, partner bid 2C, RHO doubled — raise partner's "
     "clubs with 3-card support)"),
    ("S:KJ95 H:AQ742 D:K6 C:83", "13 HCP, 5H",
     "Pass 1H Pass 2C X Pass 2S X Pass",
     "Pass (partner's X of 2S is PENALTY — with KJ95 of spades, sit and defend)"),
]

# NOTE: two example-block extensions were live A/B-tested on 2026-07-03 and
# REJECTED — do not re-add without re-measuring:
#   * anti-passivity "compete with a fit" examples (raise partner's overcall,
#     outbid a preempt): bench25 72% -> 56% combined; the model started
#     over-competing on unrelated positions (2C overcall on a flat 12,
#     pulling a penalty double).
#   * splinter + strong-2C convention examples: bench25 72% -> 68% even
#     though those sequences never occur on bench25 — any added example
#     shifts behavior globally, not just on its target pattern.

EXAMPLES_BLOCK = "Examples (study carefully):\n" + "\n".join(
    f"{hand} ({info}) | {auction} → {call}"
    for hand, info, auction, call in EXAMPLES
)

PROMPT_STYLES = ("base", "knowledge", "examples")


# NOTE: a "Legality: the auction stands at 2S — bid higher or Pass" line was
# live-tested here and REGRESSED bench25 76% -> 60%: the model over-anchored
# on it, re-opening light hands and passing competitive positions. Insufficient
# bids are rarer than that cost (FSM converts them to Pass); do not re-add
# legality text without re-measuring.


def annotate_auction(history: List[str]) -> Tuple[str, str]:
    """Annotate the auction with table roles and summarize the active seat's task.

    Roles are derived only from position in the auction relative to the active
    seat (no dealer, no hidden information): reading backward from the end, the
    calls are RHO, Partner, LHO, You, RHO, Partner, LHO, You, ... The returned
    summary tells the model whether it is opening, responding to partner,
    rebidding after its own opening, or competing.
    """
    if not history:
        return "(you are first to call)", "You are OPENING the auction."

    n = len(history)
    roles = [_ROLE_CYCLE[(n - 1 - i) % 4] for i in range(n)]
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
        elif opener_role == "You":
            summary = (
                f"YOU opened {opening_bid} earlier in this auction. You are "
                f"the OPENER choosing a rebid — do not open again."
            )
        else:
            side = "left-hand" if opener_role == "LHO" else "right-hand"
            summary = (
                f"Your {side} opponent opened {opening_bid}. You are competing/"
                f"defending, not opening."
            )
    return annotated, summary


class ContextBuilder:
    """Builds masked bidding prompts for a single seat.

    ``prompt_style`` selects the variant (see ``Config.prompt_style``):
    ``"base"`` renders only hand features + auction; ``"knowledge"`` prepends
    the SAYC reference guide; ``"examples"`` (default) additionally injects
    the kept rule blocks and the few-shot examples.
    """

    def __init__(self, prompt_style: str = "examples") -> None:
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(
                f"prompt_style must be one of {PROMPT_STYLES}, got {prompt_style!r}"
            )
        self.prompt_style = prompt_style

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
        blocks = []
        if self.prompt_style in ("knowledge", "examples"):
            blocks.append(SAYC_KNOWLEDGE)
        if self.prompt_style == "examples":
            blocks.append(KEPT_RULES)
            blocks.append(EXAMPLES_BLOCK)
        knowledge_block = "".join(f"{b}\n\n" for b in blocks)
        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"{knowledge_block}"
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
