"""Evaluation metrics (Phase 3): accuracy, contract settling, and DDS scoring.

Two layers:

* **Exact accuracy** — fraction of bids matching the expert exactly.
* **Quality (1-IMP rule)** — when a bid differs from the expert's, both auctions
  are rolled out (self-play), the settled contracts are double-dummy scored on
  the full deal, and the bid is "acceptable" if the contracts are within the
  configured threshold (default: 1 IMP via the WBF table).

The IMP table and duplicate-score formula are ported from the proven
``bridge-llm-bench`` implementation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from src.bridge import (
    SEAT_ORDER,
    normalize_call,
    parse_contract_bid,
)
from src.config import Config, DEFAULT_CONFIG
from src.schema.dataset import MockDealRecord

# A settled contract: (level, strain, declarer_seat_letter, doubled). None = passed out.
Contract = Optional[Tuple[int, str, str, int]]


# --------------------------------------------------------------------------- #
# Exact accuracy
# --------------------------------------------------------------------------- #
def calculate_accuracy(
    llm_bids: Sequence[str], expert_bids: Sequence[str]
) -> float:
    """Fraction of positions where the LLM's call equals the expert's call."""
    if not expert_bids:
        return 0.0
    correct = 0
    for llm, exp in zip(llm_bids, expert_bids):
        try:
            if normalize_call(llm) == normalize_call(exp):
                correct += 1
        except ValueError:
            continue
    return round(correct / len(expert_bids), 4)


# --------------------------------------------------------------------------- #
# WBF IMP table + duplicate scoring (ported from bridge-llm-bench)
# --------------------------------------------------------------------------- #
IMP_TABLE = [
    (20, 0), (50, 1), (90, 2), (130, 3), (170, 4),
    (220, 5), (270, 6), (320, 7), (370, 8), (430, 9),
    (500, 10), (600, 11), (750, 12), (900, 13), (1100, 14),
    (1300, 15), (1500, 16), (1750, 17), (2000, 18), (2250, 19),
    (2500, 20), (3000, 21), (3500, 22), (4000, 23),
]


def imp_diff(score_ns_1: int, score_ns_2: int) -> int:
    """IMP difference between two NS scores (positive = result 1 better)."""
    diff = score_ns_1 - score_ns_2
    abs_diff = abs(diff)
    sign = 1 if diff >= 0 else -1
    imps = 24
    for threshold, imp_val in IMP_TABLE:
        if abs_diff < threshold:
            imps = imp_val
            break
    return sign * imps


def contract_score(
    level: int, strain: str, tricks_made: int, vul: bool = False, doubled: int = 0
) -> int:
    """Duplicate score from declarer's perspective (+made / -down)."""
    tricks_needed = level + 6
    overtricks = tricks_made - tricks_needed
    if overtricks < 0:
        return _down_score(-overtricks, vul, doubled)

    trick_value = 20 if strain in ("C", "D") else 30
    base = 40 + (level - 1) * 30 if strain == "NT" else level * trick_value
    base *= (1, 2, 4)[doubled]

    game_bonus = (500 if vul else 300) if base >= 100 else 50
    slam_bonus = 0
    if level == 6:
        slam_bonus = 750 if vul else 500
    elif level == 7:
        slam_bonus = 1500 if vul else 1000
    insult = (0, 50, 100)[doubled]

    if doubled == 0:
        ot_score = overtricks * (30 if strain == "NT" else trick_value)
    elif doubled == 1:
        ot_score = overtricks * (200 if vul else 100)
    else:
        ot_score = overtricks * (400 if vul else 200)

    return base + game_bonus + slam_bonus + insult + ot_score


def _down_score(undertricks: int, vul: bool, doubled: int) -> int:
    if doubled == 0:
        return -(undertricks * (100 if vul else 50))
    mult = 2 if doubled == 2 else 1
    if vul:
        score = 200 + max(0, undertricks - 1) * 300
    else:
        if undertricks == 1:
            score = 100
        elif undertricks <= 3:
            score = 100 + (undertricks - 1) * 200
        else:
            score = 100 + 2 * 200 + (undertricks - 3) * 300
    return -(score * mult)


# --------------------------------------------------------------------------- #
# Auction -> contract
# --------------------------------------------------------------------------- #
def settle_contract(auction: List[str], dealer: str) -> Contract:
    """Determine the final contract (with declarer seat letter) from a closed auction.

    Returns ``None`` for a passed-out auction.
    """
    dealer_idx = SEAT_ORDER.index(dealer.upper())
    last: Optional[Tuple[int, str, int]] = None  # (level, strain, idx)
    doubled = 0
    for i, call in enumerate(auction):
        try:
            c = normalize_call(call)
        except ValueError:
            continue
        if c == "Pass":
            continue
        if c == "XX":
            doubled = 2
            continue
        if c == "X":
            doubled = 1
            continue
        parsed = parse_contract_bid(c)
        if parsed:
            last = (parsed[0], parsed[1], i)
            doubled = 0

    if last is None:
        return None
    level, strain, last_idx = last
    decl_letter = SEAT_ORDER[(dealer_idx + last_idx) % 4]
    decl_side = decl_letter in ("N", "S")

    # Declarer = first seat of the declaring side to name the strain.
    for i in range(last_idx + 1):
        parsed = parse_contract_bid(_safe_norm(auction[i]))
        if parsed and parsed[1] == strain:
            seat_letter = SEAT_ORDER[(dealer_idx + i) % 4]
            if (seat_letter in ("N", "S")) == decl_side:
                return (level, strain, seat_letter, doubled)
    return (level, strain, decl_letter, doubled)


def _safe_norm(call: str) -> str:
    try:
        return normalize_call(call)
    except ValueError:
        return "Pass"


# --------------------------------------------------------------------------- #
# Double-dummy scoring
# --------------------------------------------------------------------------- #
def declarer_is_vulnerable(declarer: str, vulnerability: str) -> bool:
    """Whether the declaring side is vulnerable."""
    if vulnerability == "Both":
        return True
    if vulnerability == "None":
        return False
    side = "NS" if declarer in ("N", "S") else "EW"
    return vulnerability == side


def dd_tricks(deal_pbn: str, strain: str, declarer: str) -> int:
    """Double-dummy tricks the declarer can take in ``strain`` on ``deal_pbn``."""
    from endplay.dds import calc_dd_table
    from endplay.types import Deal, Denom, Player

    strain_to_denom = {
        "C": Denom.clubs, "D": Denom.diamonds, "H": Denom.hearts,
        "S": Denom.spades, "NT": Denom.nt,
    }
    seat_to_player = {
        "N": Player.north, "E": Player.east,
        "S": Player.south, "W": Player.west,
    }
    table = calc_dd_table(Deal(deal_pbn))
    return int(table[strain_to_denom[strain], seat_to_player[declarer]])


def ns_score_for_contract(deal_pbn: str, contract: Contract, vulnerability: str) -> int:
    """NS-perspective double-dummy score for a settled contract (0 if passed out)."""
    if contract is None:
        return 0
    level, strain, declarer, doubled = contract
    tricks = dd_tricks(deal_pbn, strain, declarer)
    vul = declarer_is_vulnerable(declarer, vulnerability)
    decl_score = contract_score(level, strain, tricks, vul=vul, doubled=doubled)
    # Convert declarer-perspective to NS-perspective.
    return decl_score if declarer in ("N", "S") else -decl_score


# --------------------------------------------------------------------------- #
# The 1-IMP rule
# --------------------------------------------------------------------------- #
def format_contract(contract: Contract) -> str:
    """Human-readable contract, e.g. ``"3NT by N"``, ``"4S X by E"`` or ``"Passed out"``."""
    if contract is None:
        return "Passed out"
    level, strain, declarer, doubled = contract
    suffix = ("", " X", " XX")[doubled]
    return f"{level}{strain}{suffix} by {declarer}"


def dds_details(
    record: MockDealRecord,
    llm_bid: str,
    expert_bid: str,
    client,
    config: Config = DEFAULT_CONFIG,
) -> dict:
    """Roll out both lines and return the full comparison detail.

    Contains each line's complete auction sequence, settled contract, NS
    double-dummy score, the IMP delta, and the accept/reject verdict. This is
    the single source of truth for "the game sequence" of a position.
    """
    llm_auction = client.rollout_auction(record, llm_bid)
    expert_auction = client.rollout_auction(record, expert_bid)

    llm_contract = settle_contract(llm_auction, record.dealer)
    expert_contract = settle_contract(expert_auction, record.dealer)

    llm_ns = ns_score_for_contract(record.deal_pbn, llm_contract, record.vulnerability)
    expert_ns = ns_score_for_contract(
        record.deal_pbn, expert_contract, record.vulnerability
    )

    imp = imp_diff(llm_ns, expert_ns)

    # Orient the delta to the acting seat's side: imp/scores are NS-perspective,
    # so for an E/W seat a positive NS delta means the model's line is WORSE
    # for its own side. model_gain > 0 = model's line better for the bidder.
    seat_sign = 1 if record.seat in ("N", "S") else -1
    if config.threshold_mode == "score":
        model_gain = seat_sign * (llm_ns - expert_ns)
    else:
        model_gain = seat_sign * imp

    if config.dds_rule == "asymmetric":
        # Within threshold OR strictly better for the model's side.
        acceptable = model_gain >= -config.threshold_n
    else:
        acceptable = abs(model_gain) <= config.threshold_n

    return {
        "acceptable": acceptable,
        "dealer": record.dealer,
        "vulnerability": record.vulnerability,
        "llm_auction": llm_auction,
        "expert_auction": expert_auction,
        "llm_contract": format_contract(llm_contract),
        "expert_contract": format_contract(expert_contract),
        "llm_score_ns": llm_ns,
        "expert_score_ns": expert_ns,
        "imp_delta": imp,
        "model_gain_imp": seat_sign * imp,
    }


def evaluate_with_dds(
    record: MockDealRecord,
    llm_bid: str,
    expert_bid: str,
    client,
    config: Config = DEFAULT_CONFIG,
) -> bool:
    """Return True if ``llm_bid`` is acceptable versus ``expert_bid``.

    Exact match is trivially acceptable. Otherwise both auctions are rolled out
    via self-play, settled, double-dummy scored on the full deal, and compared
    under the configured threshold (IMP delta by default).
    """
    try:
        if normalize_call(llm_bid) == normalize_call(expert_bid):
            return True
    except ValueError:
        return False
    return dds_details(record, llm_bid, expert_bid, client, config)["acceptable"]


def classify_fallback(thinking: str) -> str:
    """Classify a :class:`BridgeBid` fallback reason from its ``thinking`` text.

    ``LocalLLMClient.get_bid`` never raises — it always falls back to
    ``Pass`` on failure, tagging *why* in ``thinking`` (see that docstring).
    This reverses the tag into a short machine-readable category so a dead
    endpoint or unparsable model output can be tallied instead of silently
    blending in with genuine ``Pass`` calls.
    """
    if thinking.startswith("transport error:"):
        return "transport_error"
    if thinking.startswith("parse error:"):
        return "parse_error"
    if thinking.startswith("illegal call"):
        return "illegal_call"
    return "none"


# --------------------------------------------------------------------------- #
# Full dataset evaluation loop (shared by CLI modes and the scenario runner)
# --------------------------------------------------------------------------- #
def evaluate_dataset(
    records: Sequence[MockDealRecord],
    client,
    config: Config = DEFAULT_CONFIG,
    use_dds: bool = True,
    detail: bool = False,
) -> dict:
    """Run the full pipeline over ``records`` and return aggregate results.

    For each record: build a masked prompt, get the model's bid, score exact
    match, and (for non-matches) apply the 1-IMP DDS rule. Returns a dict with
    per-record ``results`` plus aggregate ``accuracy`` and ``dds_acceptable_rate``.

    When ``detail`` is True, each result also carries the position context
    (``dealer``, ``vulnerability``, ``current_bidding``), the model's raw
    ``llm_thinking`` text, and, for positions that triggered a rollout, a
    ``dds`` block with both lines' full auction sequences, settled contracts
    and double-dummy scores — i.e. the game sequence per deal.

    Every result (regardless of ``detail``) carries ``fallback_reason``
    (``"none"``, ``"transport_error"``, ``"parse_error"`` or
    ``"illegal_call"``); the summary aggregates these into ``fallback_counts``
    and ``fallback_pass_rate``. A high non-``"none"`` rate means the model
    isn't actually bidding — see :func:`src.harness.llm_client.LocalLLMClient.
    verify_connection` to catch a dead endpoint before running a full dataset.
    """
    builder = client.builder
    results: List[dict] = []
    llm_bids: List[str] = []
    expert_bids: List[str] = []

    for rec in records:
        prompt = builder.build_prompt(rec)
        bid_obj = client.get_bid(prompt, history=rec.current_bidding)
        llm_bid = bid_obj.bid
        fallback_reason = classify_fallback(bid_obj.thinking)
        try:
            exact = normalize_call(llm_bid) == normalize_call(rec.expert_bid)
        except ValueError:
            exact = False

        dds_block = None
        if exact:
            acceptable = True
        elif use_dds and detail:
            # One rollout pass that yields both the verdict and the sequences.
            dds_block = dds_details(rec, llm_bid, rec.expert_bid, client, config)
            acceptable = dds_block["acceptable"]
        elif use_dds:
            acceptable = evaluate_with_dds(
                rec, llm_bid, rec.expert_bid, client, config
            )
        else:
            acceptable = False

        result = {
            "id": rec.id,
            "seat": rec.seat,
            "hcp": rec.hcp,
            "llm_bid": llm_bid,
            "expert_bid": rec.expert_bid,
            "exact_match": exact,
            "dds_acceptable": acceptable,
            "fallback_reason": fallback_reason,
        }
        if detail:
            result["dealer"] = rec.dealer
            result["vulnerability"] = rec.vulnerability
            result["current_bidding"] = list(rec.current_bidding)
            result["llm_thinking"] = bid_obj.thinking
            if dds_block is not None:
                result["dds"] = dds_block
        results.append(result)
        llm_bids.append(llm_bid)
        expert_bids.append(rec.expert_bid)

    n = len(results) or 1
    dds_rate = round(sum(1 for r in results if r["dds_acceptable"]) / n, 4)
    fallback_counts = {"none": 0, "transport_error": 0, "parse_error": 0, "illegal_call": 0}
    for r in results:
        fallback_counts[r["fallback_reason"]] += 1
    fallback_pass_rate = round((len(results) - fallback_counts["none"]) / n, 4)

    return {
        "results": results,
        "accuracy": calculate_accuracy(llm_bids, expert_bids),
        "dds_acceptable_rate": dds_rate,
        "fallback_counts": fallback_counts,
        "fallback_pass_rate": fallback_pass_rate,
        "n": len(results),
    }
