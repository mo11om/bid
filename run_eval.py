#!/usr/bin/env python
"""Bridge AI Evaluation System — CLI orchestrator.

Modes
-----
mock_data      Generate a synthetic JSONL dataset with ``redeal``.
eval_local     Evaluate a local LLM over a JSONL dataset (accuracy + DDS) and
               write an accuracy-vs-HCP report.
scenario_test  Generate constrained deals (forced N/S HCP) and evaluate them
               end-to-end.

Examples
--------
    python run_eval.py --mode mock_data --count 25
    python run_eval.py --mode eval_local --model llama3:8b --base-url http://localhost:11434/v1
    python run_eval.py --mode scenario_test --count 10 --north-hcp 12 16 --south-hcp 6 9
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from src.config import Config, VLLM_BASE_URL, OLLAMA_NATIVE_BASE_URL
from src.schema.dataset import MockDealRecord

DEFAULT_DATASET = "data/mock_eval_dataset.jsonl"
DEFAULT_REPORT = "data/accuracy_by_hcp.png"


_THINK_CHOICES = {"auto": None, "on": True, "off": False}


# --------------------------------------------------------------------------- #
def build_config(args: argparse.Namespace) -> Config:
    # If the user didn't explicitly pass --base-url, pick the right default.
    if args.base_url is None:
        base_url = VLLM_BASE_URL if args.backend == "vllm" else OLLAMA_NATIVE_BASE_URL
    else:
        base_url = args.base_url
    return Config(
        backend=args.backend,
        base_url=base_url,
        model=args.model,
        temperature=args.temperature,
        think=_THINK_CHOICES[args.think],
        prompt_style=args.prompt_style,
        retry_illegal=not args.no_retry_illegal,
        threshold_mode=args.threshold_mode,
        threshold_n=args.threshold_n,
    )


def load_dataset(path: str) -> List[MockDealRecord]:
    records: List[MockDealRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(MockDealRecord.model_validate_json(line))
    return records


def print_summary(summary: dict) -> None:
    print(f"\nPositions evaluated : {summary['n']}")
    print(f"Exact accuracy      : {summary['accuracy']:.1%}")
    print(f"DDS-acceptable rate : {summary['dds_acceptable_rate']:.1%}")
    fb = summary.get("fallback_counts")
    if fb:
        print(f"Fallback to Pass    : {summary['fallback_pass_rate']:.1%} {fb}")
        if summary["fallback_pass_rate"] > 0:
            print(
                "  (non-zero means the model isn't actually bidding for some "
                "positions — rerun with --detail to see why per position)"
            )


# --------------------------------------------------------------------------- #
def cmd_mock_data(args: argparse.Namespace) -> int:
    from src.data.mock_generator import generate_mock_dataset

    written = generate_mock_dataset(args.count, args.dataset)
    print(f"Wrote {written} records ({args.count} deals) to {args.dataset}")
    return 0


def cmd_eval_local(args: argparse.Namespace) -> int:
    from src.evaluation.metrics import evaluate_dataset
    from src.evaluation.reporter import generate_report
    from src.harness.llm_client import LocalLLMClient

    config = build_config(args)
    try:
        records = load_dataset(args.dataset)
    except FileNotFoundError:
        print(
            f"Dataset not found: {args.dataset}\n"
            f"Generate one first: python run_eval.py --mode mock_data --count 25",
            file=sys.stderr,
        )
        return 1
    if not records:
        print(f"No records found in {args.dataset}", file=sys.stderr)
        return 1

    client = LocalLLMClient(config)
    try:
        client.verify_connection()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    summary = evaluate_dataset(
        records, client, config, use_dds=not args.no_dds, detail=args.detail
    )
    print_summary(summary)

    report_path = generate_report(summary["results"], args.report)
    print(f"Report written to   : {report_path}")
    if args.results_json:
        with open(args.results_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Results JSON        : {args.results_json}")
    return 0


def cmd_scenario_test(args: argparse.Namespace) -> int:
    from src.evaluation.reporter import generate_report
    from src.simulation.scenario_runner import ScenarioRunner

    config = build_config(args)
    runner = ScenarioRunner(config=config)
    try:
        runner.client.verify_connection()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    summary = runner.run(
        args.count,
        north_hcp=tuple(args.north_hcp),
        south_hcp=tuple(args.south_hcp),
        detail=args.detail,
    )
    print_summary(summary)
    report_path = generate_report(summary["results"], args.report)
    print(f"Report written to   : {report_path}")
    if args.results_json:
        with open(args.results_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Results JSON        : {args.results_json}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bridge AI Evaluation System")
    p.add_argument(
        "--mode",
        required=True,
        choices=["mock_data", "eval_local", "scenario_test"],
    )
    p.add_argument("--count", type=int, default=25, help="Number of deals.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="JSONL dataset path.")
    p.add_argument("--report", default=DEFAULT_REPORT, help="Report PNG path.")
    p.add_argument("--results-json", default=None, help="Optional results JSON dump.")

    # Model / endpoint
    p.add_argument(
        "--backend",
        choices=["ollama", "vllm"],
        default="ollama",
        help=(
            "LLM backend. 'ollama' uses the native /api/chat endpoint (correctly "
            "honours think:false). 'vllm' uses the OpenAI-compatible /v1/chat/completions "
            "endpoint. Default: ollama."
        ),
    )
    p.add_argument("--base-url", default=None, help="Override endpoint URL.")
    p.add_argument("--model", default=Config.model, help="Model name (run 'ollama list' for exact tags).")
    p.add_argument("--temperature", type=float, default=Config.temperature)
    p.add_argument(
        "--think",
        choices=list(_THINK_CHOICES),
        default="auto",
        help=(
            "Reasoning/thinking mode for servers that support it (Ollama with "
            "qwen3/deepseek-r1/gpt-oss, etc.). 'auto' omits the parameter and "
            "lets the server/model decide; 'on'/'off' force it. Ignored by "
            "servers that don't recognize the field."
        ),
    )

    # Threshold
    p.add_argument(
        "--prompt-style",
        choices=["base", "knowledge", "examples"],
        default=Config.prompt_style,
        help=(
            "Prompt variant: 'base' = hand features + auction only; "
            "'knowledge' = + SAYC reference guide; 'examples' (default) = "
            "+ targeted rules and few-shot examples (ablation-proven recipe)."
        ),
    )
    p.add_argument(
        "--no-retry-illegal",
        action="store_true",
        help=(
            "Disable the one corrective re-ask after an FSM-rejected call "
            "(illegal calls then fall straight back to Pass, which can be "
            "accidentally 'correct')."
        ),
    )
    p.add_argument("--threshold-mode", choices=["imp", "score"], default="imp")
    p.add_argument("--threshold-n", type=int, default=1)
    p.add_argument("--no-dds", action="store_true", help="Skip DDS (exact match only).")
    p.add_argument(
        "--detail",
        action="store_true",
        help=(
            "Include each position's full game sequence in the results: dealer, "
            "vulnerability, the auction shown to the model, and (for rolled-out "
            "positions) both lines' complete auctions, contracts and DD scores. "
            "Best combined with --results-json."
        ),
    )

    # Scenario constraints
    p.add_argument("--north-hcp", type=int, nargs=2, default=[0, 40], metavar=("LO", "HI"))
    p.add_argument("--south-hcp", type=int, nargs=2, default=[0, 40], metavar=("LO", "HI"))
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "mock_data": cmd_mock_data,
        "eval_local": cmd_eval_local,
        "scenario_test": cmd_scenario_test,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
