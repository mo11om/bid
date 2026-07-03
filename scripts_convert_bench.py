"""Convert bridge-llm-bench datasets into the bid evaluator's JSONL schema,
using the **Ben SAYC** column as the reference (expert) call so the two
projects share an oracle.

Two sources:

* ``--set 25``  (default) — ``results/oracle_comparison.csv`` (25 OpenSpiel
  hands) → ``data/bench25_bensayc.jsonl``.
* ``--set 150`` — ``data/ben_sayc_100.csv`` (150 positions over 14 deals)
  → ``data/bench150_bensayc.jsonl``.

Limitation: the benchmark CSVs have only ONE hand per row (no full deal), so
``deal_pbn``/``all_hands`` are left empty and DDS quality scoring cannot run —
these datasets are exact-match only (run with --no-dds).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.bridge import count_hcp, compute_shape, compute_ltc, seat_to_act

BENCH_ROOT = Path("../bridge-llm-bench")
SOURCES = {
    "25": (BENCH_ROOT / "results/oracle_comparison.csv", "ben_sayc",
           Path("data/bench25_bensayc.jsonl")),
    "150": (BENCH_ROOT / "data/ben_sayc_100.csv", "ben_sayc_bid",
            Path("data/bench150_bensayc.jsonl")),
}


def to_dotted(hand: str) -> str:
    """'S:Q52 H:6543 D:K732 C:AJ' -> 'S:Q52.H:6543.D:K732.C:AJ'."""
    return ".".join(hand.split())


def is_clean(v: str) -> bool:
    return bool(v) and not v.startswith("[")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SOURCES), default="25",
                    help="Which benchmark set to convert (default: 25).")
    args = ap.parse_args()
    src, oracle_col, out = SOURCES[args.set]

    rows = list(csv.DictReader(open(src)))
    written = 0
    with open(out, "w") as fh:
        for i, r in enumerate(rows):
            oracle = (r[oracle_col] or "").strip()
            if not is_clean(oracle):
                continue
            hand = to_dotted(r["hand"])
            auction = r["auction"].split() if r["auction"].strip() else []
            # Dealer unknown in the CSV; assume N. Seat is inferred by rotation.
            dealer = "N"
            seat = seat_to_act(dealer, len(auction))
            rec = {
                "id": f"bench-{r.get('index', i)}",
                "seat": seat,
                "hand": hand,
                "hcp": count_hcp(hand),
                "shape": compute_shape(hand),
                "ltc": compute_ltc(hand),
                "current_bidding": auction,
                "expert_bid": oracle,
                # full-deal fields unknown -> DDS disabled for this set
                "deal_pbn": "",
                "all_hands": {},
                "dealer": dealer,
                "vulnerability": "None",
            }
            fh.write(json.dumps(rec) + "\n")
            written += 1
    print(f"Wrote {written} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
