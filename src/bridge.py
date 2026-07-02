"""Pure bridge primitives shared across the system.

No I/O, no third-party deps — just hand/bid parsing and the standard hand
evaluations (HCP, shape, Losing Trick Count). Kept dependency-free so it is
trivially unit-testable offline (no redeal / endplay / network required).

Hand string formats accepted by :func:`parse_hand`:

* Labeled (canonical in datasets): ``"S:AK7.H:QJ3.D:854.C:KT62"``
* Plain PBN holding order S.H.D.C:        ``"AK7.QJ3.854.KT62"``

Void suits are the empty string between dots, e.g. ``"S:.H:AKQ..."``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

SUITS: Tuple[str, ...] = ("S", "H", "D", "C")
STRAINS: Tuple[str, ...] = ("C", "D", "H", "S", "NT")  # ascending rank order
RANK_ORDER = "AKQJT98765432"  # high -> low

HCP_VALUES = {"A": 4, "K": 3, "Q": 2, "J": 1}


# --------------------------------------------------------------------------- #
# Hand parsing
# --------------------------------------------------------------------------- #
def parse_hand(hand_str: str) -> Dict[str, str]:
    """Parse a hand string into ``{suit: ranks}`` (ranks high->low, may be "").

    Accepts both the labeled (``S:..H:..``) and plain (``..``) PBN forms.
    """
    s = hand_str.strip()
    if not s:
        raise ValueError("empty hand string")

    holdings: Dict[str, str] = {suit: "" for suit in SUITS}

    if ":" in s:
        # Labeled form: split on '.' between suit blocks like 'S:AK7'.
        for block in s.split("."):
            block = block.strip()
            if not block or ":" not in block:
                continue
            label, ranks = block.split(":", 1)
            label = label.strip().upper()
            if label in holdings:
                holdings[label] = _normalize_ranks(ranks)
    else:
        # Plain PBN: four dot-separated holdings in S.H.D.C order.
        parts = s.split(".")
        if len(parts) != 4:
            raise ValueError(f"plain PBN hand must have 4 suits, got: {hand_str!r}")
        for suit, ranks in zip(SUITS, parts):
            holdings[suit] = _normalize_ranks(ranks)

    return holdings


def _normalize_ranks(ranks: str) -> str:
    """Uppercase, strip spaces, sort ranks high->low; map '10' to 'T'."""
    cleaned = ranks.strip().upper().replace("10", "T").replace(" ", "")
    return "".join(sorted(cleaned, key=RANK_ORDER.index))


def to_pbn_hand(hand_str: str) -> str:
    """Convert any accepted hand string to plain PBN holdings ``S.H.D.C``."""
    h = parse_hand(hand_str)
    return ".".join(h[suit] for suit in SUITS)


def to_labeled_hand(hand_str: str) -> str:
    """Convert any accepted hand string to the labeled canonical form."""
    h = parse_hand(hand_str)
    return ".".join(f"{suit}:{h[suit]}" for suit in SUITS)


# --------------------------------------------------------------------------- #
# Hand evaluation
# --------------------------------------------------------------------------- #
def count_hcp(hand: str | Dict[str, str]) -> int:
    """High-card points: A=4, K=3, Q=2, J=1."""
    holdings = hand if isinstance(hand, dict) else parse_hand(hand)
    total = 0
    for ranks in holdings.values():
        for card in ranks:
            total += HCP_VALUES.get(card, 0)
    return total


def suit_lengths(hand: str | Dict[str, str]) -> Dict[str, int]:
    """Length of each suit, keyed by suit letter."""
    holdings = hand if isinstance(hand, dict) else parse_hand(hand)
    return {suit: len(holdings[suit]) for suit in SUITS}


def compute_shape(hand: str | Dict[str, str]) -> str:
    """Distribution as a sorted-descending string, e.g. ``"4-3-3-3"``."""
    lengths = sorted(suit_lengths(hand).values(), reverse=True)
    return "-".join(str(n) for n in lengths)


def compute_ltc(hand: str | Dict[str, str]) -> int:
    """Standard Losing Trick Count.

    For each suit, consider the top three card slots (or fewer if the suit is
    short). Each of the Ace, King and Queen *missing* from those slots is a
    loser. A void has zero losers; a singleton can lose at most one; a doubleton
    at most two.
    """
    holdings = hand if isinstance(hand, dict) else parse_hand(hand)
    losers = 0
    for ranks in holdings.values():
        length = len(ranks)
        slots = min(3, length)
        if slots == 0:
            continue
        held_top = set(ranks[:slots]) & {"A", "K", "Q"}
        # Only honors that *fit* in the available slots count toward losers.
        # With 1 slot only the Ace matters; with 2 slots A,K; with 3 slots A,K,Q.
        considered = {1: {"A"}, 2: {"A", "K"}, 3: {"A", "K", "Q"}}[slots]
        missing = considered - held_top
        losers += len(missing)
    return losers


def classify_shape(shape: str) -> str:
    """Coarse hand-type label from a ``compute_shape`` string, e.g. ``"5-3-3-2"``.

    Returns ``"Balanced"`` (4-3-3-3, 4-4-3-2, 5-3-3-2), ``"Two-suited"`` (two
    longest suits are 5+ and 4+), else ``"Unbalanced"`` (single- and
    three-suiters, freakish shapes).
    """
    if shape in ("4-3-3-3", "4-4-3-2", "5-3-3-2"):
        return "Balanced"
    lengths = [int(n) for n in shape.split("-")]
    if len(lengths) >= 2 and lengths[0] >= 5 and lengths[1] >= 4:
        return "Two-suited"
    return "Unbalanced"


# --------------------------------------------------------------------------- #
# Bid parsing
# --------------------------------------------------------------------------- #
def normalize_call(call: str) -> str:
    """Normalize a single call to canonical form.

    Returns one of ``"Pass"``, ``"X"``, ``"XX"`` or a contract bid like ``"3NT"``.
    Raises ``ValueError`` for anything unrecognized.
    """
    c = call.strip().upper().replace(" ", "")
    if c in ("PASS", "P", "-"):
        return "Pass"
    if c in ("X", "DBL", "DOUBLE"):
        return "X"
    if c in ("XX", "RDBL", "REDOUBLE"):
        return "XX"
    if len(c) >= 2 and c[0] in "1234567":
        level = c[0]
        strain = c[1:]
        if strain == "N":
            strain = "NT"
        if strain in STRAINS:
            return f"{level}{strain}"
    raise ValueError(f"unrecognized call: {call!r}")


def is_contract_bid(call: str) -> bool:
    """True for a level+strain bid (not Pass/X/XX)."""
    try:
        c = normalize_call(call)
    except ValueError:
        return False
    return c not in ("Pass", "X", "XX")


def parse_contract_bid(call: str) -> Optional[Tuple[int, str]]:
    """Return ``(level, strain)`` for a contract bid, else ``None``."""
    if not is_contract_bid(call):
        return None
    c = normalize_call(call)
    return int(c[0]), c[1:]


def bid_rank(call: str) -> int:
    """Strictly-increasing rank for contract bids (1C=0 .. 7NT=34).

    Non-contract calls return -1.
    """
    parsed = parse_contract_bid(call)
    if parsed is None:
        return -1
    level, strain = parsed
    return (level - 1) * 5 + STRAINS.index(strain)


# --------------------------------------------------------------------------- #
# Auction state
# --------------------------------------------------------------------------- #
SEAT_ORDER: Tuple[str, ...] = ("N", "E", "S", "W")


def seat_to_act(dealer: str, num_calls: int) -> str:
    """Seat whose turn it is after ``num_calls`` calls, given the dealer."""
    start = SEAT_ORDER.index(dealer.upper())
    return SEAT_ORDER[(start + num_calls) % 4]


def auction_is_closed(calls: List[str]) -> bool:
    """True once the auction has ended.

    * Four opening passes -> passed out.
    * Otherwise, three passes following the last contract bid (or its
      double/redouble) close the auction.
    """
    norm: List[str] = []
    for c in calls:
        try:
            norm.append(normalize_call(c))
        except ValueError:
            return False
    if len(norm) < 4:
        return False
    if not any(is_contract_bid(c) for c in norm):
        return all(c == "Pass" for c in norm[:4])
    return norm[-3:] == ["Pass", "Pass", "Pass"]


# --------------------------------------------------------------------------- #
# HCP bucketing (for reporting)
# --------------------------------------------------------------------------- #
def hcp_bucket(hcp: int) -> str:
    """Bucket HCP into the report ranges 0-10, 11-15, 16+."""
    if hcp <= 10:
        return "0-10"
    if hcp <= 15:
        return "11-15"
    return "16+"


HCP_BUCKETS: List[str] = ["0-10", "11-15", "16+"]
