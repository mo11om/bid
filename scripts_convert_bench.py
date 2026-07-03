"""Convert bridge-llm-bench datasets into the bid evaluator's JSONL schema,
using the **Ben SAYC** column as the reference (expert) call so the two
projects share an oracle.

Two sources:

* ``--set 25``  (default) — ``results/oracle_comparison.csv`` (25 OpenSpiel
  hands) → ``data/bench25_bensayc.jsonl``.
* ``--set 150`` — ``data/ben_sayc_100.csv`` (150 positions over 14 deals)
  → ``data/bench150_bensayc.jsonl``.

Full-deal reconstruction: the CSVs list positions deal by deal, rotating
through the four seats — so each deal's first four rows contain all four
hands. Those are reassembled into ``all_hands``/``deal_pbn`` (validated to 52
unique cards), which unlocks the harness's rollout + double-dummy quality
scoring. Dealer is assumed N (not recorded in the CSVs) and vulnerability
None, consistent with the seat rotation used for ``seat``.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.bridge import (
    compute_ltc,
    compute_shape,
    count_hcp,
    parse_hand,
    seat_to_act,
    to_pbn_hand,
)
from src.data.mock_generator import _deal_to_full_pbn

BENCH_ROOT = Path("../bridge-llm-bench")
SOURCES = {
    "25": (BENCH_ROOT / "results/oracle_comparison.csv", "ben_sayc",
           Path("data/bench25_bensayc.jsonl")),
    "150": (BENCH_ROOT / "data/ben_sayc_100.csv", "ben_sayc_bid",
            Path("data/bench150_bensayc.jsonl")),
}
SEATS = ("N", "E", "S", "W")


def to_dotted(hand: str) -> str:
    """'S:Q52 H:6543 D:K732 C:AJ' -> 'S:Q52.H:6543.D:K732.C:AJ'."""
    return ".".join(hand.split())


def is_clean(v: str) -> bool:
    return bool(v) and not v.startswith("[")


def group_deals(rows: list[dict]) -> list[list[dict]]:
    """Split the row stream into deals; a new deal starts at an empty auction."""
    deals: list[list[dict]] = []
    for r in rows:
        if not r["auction"].strip():
            deals.append([])
        if not deals:
            raise SystemExit("first CSV row does not start a deal (auction non-empty)")
        deals[-1].append(r)
    return deals


def reconstruct_hands(deal_rows: list[dict], deal_idx: int) -> dict[str, str] | None:
    """Rebuild {seat: labeled hand} from a deal's first four rows.

    Row j of a deal has an auction of length j, so with dealer N its acting
    seat is SEATS[j % 4] — the first four rows cover N, E, S, W exactly.
    Validates the reassembled deal holds 52 unique cards (13 per seat).

    Returns ``None`` for a deal with fewer than four rows (the 25-row CSV is
    truncated mid-deal): those records stay exact-match only, with empty
    ``deal_pbn``/``all_hands``.
    """
    if len(deal_rows) < 4:
        print(f"warning: deal {deal_idx} has only {len(deal_rows)} row(s) — "
              f"cannot reconstruct hands; DDS disabled for these records",
              file=sys.stderr)
        return None
    labeled = {SEATS[j]: to_dotted(deal_rows[j]["hand"]) for j in range(4)}
    cards = set()
    for seat, hand in labeled.items():
        holdings = parse_hand(hand)
        n = sum(len(v) for v in holdings.values())
        if n != 13:
            raise SystemExit(f"deal {deal_idx}: seat {seat} has {n} cards")
        cards.update((s, c) for s, v in holdings.items() for c in v)
    if len(cards) != 52:
        raise SystemExit(f"deal {deal_idx}: {len(cards)} unique cards, expected 52")
    return labeled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SOURCES), default="25",
                    help="Which benchmark set to convert (default: 25).")
    args = ap.parse_args()
    src, oracle_col, out = SOURCES[args.set]

    rows = list(csv.DictReader(open(src)))
    deals = group_deals(rows)
    written = 0
    with open(out, "w") as fh:
        i = -1
        for deal_idx, deal_rows in enumerate(deals):
            labeled = reconstruct_hands(deal_rows, deal_idx)
            if labeled is not None:
                all_hands = {s: to_pbn_hand(h) for s, h in labeled.items()}
                deal_pbn = _deal_to_full_pbn(labeled)
            else:
                all_hands, deal_pbn = {}, ""
            for r in deal_rows:
                i += 1
                oracle = (r[oracle_col] or "").strip()
                if not is_clean(oracle):
                    continue
                hand = to_dotted(r["hand"])
                auction = r["auction"].split() if r["auction"].strip() else []
                # Dealer unknown in the CSV; assume N. Seat inferred by rotation.
                dealer = "N"
                seat = seat_to_act(dealer, len(auction))
                if all_hands and to_pbn_hand(hand) != all_hands[seat]:
                    raise SystemExit(
                        f"deal {deal_idx}: row hand does not match seat {seat} "
                        f"of the reconstructed deal"
                    )
                rec = {
                    "id": f"bench-{r.get('index', i)}",
                    "seat": seat,
                    "hand": hand,
                    "hcp": count_hcp(hand),
                    "shape": compute_shape(hand),
                    "ltc": compute_ltc(hand),
                    "current_bidding": auction,
                    "expert_bid": oracle,
                    "deal_pbn": deal_pbn,
                    "all_hands": all_hands,
                    "dealer": dealer,
                    "vulnerability": "None",
                }
                fh.write(json.dumps(rec) + "\n")
                written += 1
    print(f"Wrote {written} records to {out} ({len(deals)} deals reconstructed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
