"""Majority-vote evaluation replicating bridge-llm-bench's winning recipe:
tuned SAYC-knowledge prompt + k-sample majority vote at temperature t.

Reuses the bid harness (ContextBuilder + LocalLLMClient parsing) but bypasses
the response cache so each of the k samples is independent. Exact-match only
(the bench25 set has no full deals for DDS).

Usage:
    python scripts_vote_eval.py --k 9 --temp 0.5 --prompt-style examples
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.config import Config, OLLAMA_NATIVE_BASE_URL
from src.bridge import normalize_call
from src.harness.prompt_builder import ContextBuilder
from src.harness.llm_client import LocalLLMClient
from src.schema.dataset import MockDealRecord


class NullCache:
    """No-op cache so repeated identical prompts yield independent samples."""
    def get(self, *a, **k): return None
    def set(self, *a, **k): pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/bench25_bensayc.jsonl")
    ap.add_argument("--model", default="gemma4:26b")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--prompt-style", choices=["base", "knowledge", "examples"],
                    default="examples")
    ap.add_argument("--out", default="data/bench25_gemma4_26b_vote.json")
    args = ap.parse_args()

    cfg = Config(backend="ollama", base_url=OLLAMA_NATIVE_BASE_URL,
                 model=args.model, temperature=args.temp, think=False,
                 prompt_style=args.prompt_style)
    builder = ContextBuilder(prompt_style=args.prompt_style)
    client = LocalLLMClient(cfg, builder=builder, cache=NullCache())
    client.verify_connection()

    records = [MockDealRecord.model_validate_json(l)
               for l in open(args.dataset) if l.strip()]
    print(f"prompt_style={args.prompt_style}  k={args.k}  temp={args.temp}  "
          f"n={len(records)}  model={args.model}", flush=True)

    results, correct = [], 0
    for rec in records:
        prompt = builder.build_prompt(rec)
        votes = []
        for _ in range(args.k):
            bid = client.get_bid(prompt, history=list(rec.current_bidding))
            try:
                votes.append(normalize_call(bid.bid))
            except ValueError:
                pass
        if not votes:
            winner = "Pass"
        else:
            top = Counter(votes).most_common()
            best = top[0][1]
            # tie-break: among the most-voted, keep first-seen order
            winner = next(v for v, c in top if c == best)
        try:
            exact = normalize_call(winner) == normalize_call(rec.expert_bid)
        except ValueError:
            exact = False
        correct += exact
        results.append({"id": rec.id, "seat": rec.seat,
                        "vote_bid": winner, "expert_bid": rec.expert_bid,
                        "exact_match": exact, "votes": Counter(votes)})
        print(f"  {rec.id:10} {'✓' if exact else '✗'} vote={winner:5} "
              f"oracle={rec.expert_bid:5} dist={dict(Counter(votes))}", flush=True)

    acc = correct / len(records)
    print(f"\nExact accuracy (vote@{args.k}, t={args.temp}): "
          f"{correct}/{len(records)} = {acc:.1%}", flush=True)
    json.dump({"accuracy": acc, "k": args.k, "temp": args.temp,
               "n": len(records),
               "results": [{**r, "votes": dict(r["votes"])} for r in results]},
              open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
