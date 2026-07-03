"""Situation-triggered prompt blocks (masked-info only).

Detects bidding situations from the active seat's OWN hand plus the public
auction, and supplies a targeted guidance block per situation. The point of
the mechanism: four separate live A/Bs showed that *globally* added prompt
text regresses this model on unrelated positions (docs/SESSION_LOG.md §12–13).
Injecting a block only when its situation fires confines the blast radius —
every non-fired position renders a byte-identical prompt, so non-regression
outside the target class holds by construction.

Detection consumes ONLY (hand, hcp, history) — the same masked inputs the
prompt itself uses — so the masking guarantee is untouched.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.bridge import is_contract_bid, parse_contract_bid, parse_hand

# Mirrors prompt_builder._ROLE_CYCLE (kept local to avoid a circular import):
# back-counted from the active seat, the auction rotates RHO, Partner, LHO,
# You with period 4.
_ROLE_CYCLE = ("RHO", "Partner", "LHO", "You")

_TOP_HONORS = set("AKQJ")

# Stable evaluation/injection order (by measured IMP cost of the theme).
ALL_TAGS = (
    "double_candidate",
    "partner_game_drive",
    "nt_game_candidate",
    "strong_2c",
)


def _roles(history: List[str]) -> List[str]:
    n = len(history)
    return [_ROLE_CYCLE[(n - 1 - i) % 4] for i in range(n)]


def _first_bid(history: List[str]) -> Optional[int]:
    """Index of the opening (first non-Pass) call, or None."""
    return next((i for i, c in enumerate(history) if c != "Pass"), None)


def _last_nonpass(history: List[str]) -> Optional[int]:
    return next(
        (i for i in range(len(history) - 1, -1, -1) if history[i] != "Pass"), None
    )


def _suit_stack(hand: str, strain: str) -> bool:
    """4+ cards in ``strain`` including 2+ of the top four honors."""
    if strain == "NT":
        return False
    ranks = parse_hand(hand).get(strain, "")
    return len(ranks) >= 4 and sum(r in _TOP_HONORS for r in ranks) >= 2


def _stopped(hand: str, strain: str) -> bool:
    """Same stopper notion as HandFacts: A, guarded K, or Qxx."""
    if strain == "NT":
        return True
    ranks = parse_hand(hand).get(strain, "")
    return (
        "A" in ranks
        or ("K" in ranks and len(ranks) >= 2)
        or ("Q" in ranks and len(ranks) >= 3)
    )


# --------------------------------------------------------------------------- #
# Per-tag predicates
# --------------------------------------------------------------------------- #
def _double_candidate(hand: str, hcp: int, history: List[str], roles: List[str]) -> bool:
    """Opponents' undoubled contract bid stands, and we have double material.

    Fires when the last non-Pass call is an opponent's contract bid AND any of:
    15+ HCP; their suit stacked (4+ cards, 2 of AKQJ); or we sit in the
    balancing (pass-out) seat behind two passes.
    """
    idx = _last_nonpass(history)
    if idx is None or not is_contract_bid(history[idx]):
        return False  # nothing to double, or it's already X/XX
    if roles[idx] not in ("RHO", "LHO"):
        return False
    level, strain = parse_contract_bid(history[idx])  # type: ignore[misc]
    balancing = len(history) - 1 - idx == 2  # their bid, then Pass, Pass
    return hcp >= 15 or balancing or _suit_stack(hand, strain)


def _partner_game_drive(hand: str, hcp: int, history: List[str], roles: List[str]) -> bool:
    """Partner has shown a hand we tend to park under.

    (a) Partner made a takeout X of the opponents' opening and we hold a
        4+ card major, 13+ HCP, or a 6+ suit.
    (b) Splinter context: our side opened 1H/1S and the other partner jumped
        to 4C/4D (never natural).
    (c) We opened, the opponents competed (X or a bid), and partner freely
        bid a suit in which we hold 4+ cards — a fit worth driving, not
        parking (the bench-29/33/37 family).
    """
    first = _first_bid(history)
    if first is None:
        return False
    lengths = {s: len(r) for s, r in parse_hand(hand).items()}

    # (a) partner's takeout X of their opening
    if roles[first] in ("RHO", "LHO"):
        partner_x = any(
            c == "X" and roles[i] == "Partner" for i, c in enumerate(history)
        )
        if partner_x and (
            lengths["S"] >= 4 or lengths["H"] >= 4 or hcp >= 13
            or max(lengths.values()) >= 6
        ):
            return True

    # (b) splinter over our 1M — either partner splintered over our opening,
    # or we splintered and partner is continuing.
    if history[first] in ("1H", "1S") and roles[first] in ("You", "Partner"):
        other = "Partner" if roles[first] == "You" else "You"
        if any(
            c in ("4C", "4D") and roles[i] == other
            for i, c in enumerate(history[first + 1:], start=first + 1)
        ):
            return True

    # (c) we opened, they competed, partner bid a suit we support (4+ cards)
    if roles[first] == "You":
        they_competed = any(
            (c == "X" or is_contract_bid(c)) and roles[i] in ("RHO", "LHO")
            for i, c in enumerate(history)
        )
        if they_competed:
            for i, c in enumerate(history):
                if roles[i] == "Partner" and is_contract_bid(c):
                    strain = parse_contract_bid(c)[1]  # type: ignore[index]
                    if strain != "NT" and lengths[strain] >= 4:
                        return True
    return False


def _nt_game_candidate(hand: str, hcp: int, history: List[str], roles: List[str]) -> bool:
    """Our side showed opening values; we hold 12+ HCP with every opponent
    suit stopped — game values, prone to stalling in a partscore."""
    if hcp < 12:
        return False
    our_one_level = any(
        is_contract_bid(c)
        and parse_contract_bid(c)[0] == 1  # type: ignore[index]
        and roles[i] in ("You", "Partner")
        for i, c in enumerate(history)
    )
    if not our_one_level:
        return False
    our_suits = {
        parse_contract_bid(c)[1]  # type: ignore[index]
        for i, c in enumerate(history)
        if is_contract_bid(c) and roles[i] in ("You", "Partner")
    }
    # A suit our side bid first doesn't need our stopper (their bid of it is
    # a raise/cue, not a suit to run against us).
    opp_suits = {
        parse_contract_bid(c)[1]  # type: ignore[index]
        for i, c in enumerate(history)
        if is_contract_bid(c) and roles[i] in ("RHO", "LHO")
    } - {"NT"} - our_suits
    if not opp_suits:
        return False
    return all(_stopped(hand, s) for s in opp_suits)


def _strong_2c(hand: str, hcp: int, history: List[str], roles: List[str]) -> bool:
    """Our side opened a strong, game-forcing 2C."""
    first = _first_bid(history)
    return (
        first is not None
        and history[first] == "2C"
        and roles[first] in ("You", "Partner")
    )


_PREDICATES = {
    "double_candidate": _double_candidate,
    "partner_game_drive": _partner_game_drive,
    "nt_game_candidate": _nt_game_candidate,
    "strong_2c": _strong_2c,
}


def detect_situations(hand: str, hcp: int, history: List[str]) -> List[str]:
    """Fired situation tags for this (masked) position, in ALL_TAGS order."""
    if not history:
        return []
    roles = _roles(history)
    return [t for t in ALL_TAGS if _PREDICATES[t](hand, hcp, history, roles)]


# --------------------------------------------------------------------------- #
# Injected guidance blocks (generic content; example hands are fresh and
# covered by the anti-leakage fence in tests/test_prompt_examples.py).
# --------------------------------------------------------------------------- #
SITUATION_BLOCKS: Dict[str, str] = {
    "double_candidate": (
        "DOUBLE CHECK — a double may be your best call here:\n"
        "- PENALTY X: the opponents freely bid to a contract and you hold "
        "their suit strongly (4+ cards with honors) or 15+ HCP — double "
        "rather than pass quietly.\n"
        "- BALANCING X: their low-level bid is followed by two passes; in the "
        "pass-out seat reopen with X on 8+ HCP rather than sell out.\n"
        "- RESPONSIVE X: partner doubled and RHO raised; X shows values with "
        "no clear suit.\n"
        "Example: S:A4 H:KQT7 D:J973 C:K82 (13 HCP) | 1H Pass 2H Pass Pass → "
        "X (pass-out seat with hearts behind them — do NOT sell out to 2H)."
    ),
    "partner_game_drive": (
        "PARTNER SHOWED STRENGTH — do not park below game:\n"
        "- A jump to 4C/4D over our 1H/1S opening is a SPLINTER: shortness "
        "there, 4+ trumps, game-forcing. It is never natural — sign off in 4 "
        "of the major with a minimum, cue-bid or 4NT with extras.\n"
        "- After partner's takeout X of their opening, bid your 4-card major "
        "at the level needed; over their raise, compete to the 4 level with "
        "a fit.\n"
        "- When partner freely bid a suit you hold 4+ cards in, you have a "
        "FIT: raise or return to partner's suit at the level the auction "
        "demands rather than passing.\n"
        "Example: S:K862 H:A54 D:J83 C:T93 (8 HCP, 4S) | Pass 1D X 4D → 4S "
        "(partner's takeout X promises support — outbid their raise)."
    ),
    "nt_game_candidate": (
        "GAME CHECK (notrump): your side has opening values and you hold 12+ "
        "HCP with every opponent suit stopped — the combined values are "
        "enough for GAME. Bid 3NT (or raise to game) instead of stalling in "
        "2NT or a partscore; stopping low wastes the game bonus.\n"
        "Example: S:KJ4 H:Q73 D:AT92 C:K85 (13 HCP, bal) | 1C 1S → 3NT "
        "(partner opened, spades stopped — bid the game, not 2NT)."
    ),
    "strong_2c": (
        "STRONG 2C AUCTION: our side's 2C opening shows 22+ HCP or a "
        "game-forcing hand. The auction is FORCING — do not pass below game. "
        "Responder keeps bidding (raise with 3+ support, cheapest bid with "
        "nothing); opener keeps describing; with trump agreement and slam "
        "interest use Blackwood 4NT.\n"
        "Example: S:864 H:J752 D:T93 C:942 (1 HCP) | 2C Pass 2D Pass 2S Pass "
        "→ 3S (2C is game-forcing — raise partner's spades even with nothing, "
        "do NOT pass)."
    ),
}


def parse_situations_setting(setting: str) -> Tuple[str, ...]:
    """Validate/expand a Config.situations value into a tag tuple."""
    s = setting.strip().lower()
    if s == "all":
        return ALL_TAGS
    if s == "none":
        return ()
    tags = tuple(t.strip() for t in s.split(",") if t.strip())
    bad = [t for t in tags if t not in ALL_TAGS]
    if bad:
        raise ValueError(
            f"unknown situation tag(s) {bad}; valid: {list(ALL_TAGS)} or 'all'/'none'"
        )
    return tags
