"""Evaluation loop + reporter tests using a stub client (no network/endplay)."""

from src.evaluation.metrics import classify_fallback, evaluate_dataset
from src.evaluation.reporter import aggregate_by_bucket, generate_report
from src.harness.prompt_builder import ContextBuilder
from src.schema.dataset import BridgeBid
from src.data.mock_generator import records_from_labeled

LABELED = {
    "N": "S:AK7.H:QJ3.D:854.C:KT62",   # 13 HCP -> expert opens 1NT
    "E": "S:432.H:432.D:432.C:5432",
    "S": "S:T98.H:T98.D:T98.C:T987",
    "W": "S:765.H:765.D:765.C:6543",
}


class FixedClient:
    """Stub: always returns the same call; never rolls out an auction."""

    def __init__(self, bid: str = "Pass") -> None:
        self.builder = ContextBuilder()
        self._bid = bid

    def get_bid(self, prompt, history=None):
        return BridgeBid(thinking="stub", bid=self._bid)

    def rollout_auction(self, record, first_bid):  # pragma: no cover - unused here
        raise AssertionError("rollout should not be called with use_dds=False")


def test_evaluate_dataset_exact_only():
    records = records_from_labeled(LABELED, board_number=1)
    client = FixedClient(bid="Pass")  # matches 3 of 4 experts (all but N's 1NT)
    summary = evaluate_dataset(records, client, use_dds=False)

    assert summary["n"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["dds_acceptable_rate"] == 0.75  # N non-match, dds off -> not acceptable
    n_record = next(r for r in summary["results"] if r["seat"] == "N")
    assert n_record["exact_match"] is False
    # No fallback occurred (the stub always returns a clean BridgeBid).
    assert n_record["fallback_reason"] == "none"
    assert summary["fallback_counts"] == {
        "none": 4, "transport_error": 0, "parse_error": 0, "illegal_call": 0,
    }
    assert summary["fallback_pass_rate"] == 0.0


def test_classify_fallback():
    assert classify_fallback("transport error: connection refused") == "transport_error"
    assert classify_fallback("parse error: invalid json") == "parse_error"
    assert classify_fallback("illegal call 'Pass' -> Pass") == "illegal_call"
    assert classify_fallback("13 HCP, open 1NT") == "none"


def test_evaluate_dataset_tallies_fallbacks():
    class FlakyClient:
        """Every get_bid call fails transport; rollout is never reached (use_dds=False)."""

        def __init__(self) -> None:
            self.builder = ContextBuilder()

        def get_bid(self, prompt, history=None):
            return BridgeBid(thinking="transport error: 404 model not found", bid="Pass")

    records = records_from_labeled(LABELED, board_number=1)
    summary = evaluate_dataset(records, FlakyClient(), use_dds=False)

    assert summary["fallback_counts"]["transport_error"] == 4
    assert summary["fallback_pass_rate"] == 1.0
    assert all(r["fallback_reason"] == "transport_error" for r in summary["results"])


def test_aggregate_and_report(tmp_path):
    eval_results = [
        {"hcp": 5, "exact_match": True, "dds_acceptable": True},
        {"hcp": 8, "exact_match": False, "dds_acceptable": True},
        {"hcp": 13, "exact_match": True, "dds_acceptable": True},
        {"hcp": 18, "exact_match": False, "dds_acceptable": False},
    ]
    agg = aggregate_by_bucket(eval_results)
    assert agg["0-10"]["n"] == 2
    assert agg["0-10"]["exact_accuracy"] == 0.5
    assert agg["0-10"]["dds_accuracy"] == 1.0
    assert agg["16+"]["dds_accuracy"] == 0.0

    out = tmp_path / "report.png"
    path = generate_report(eval_results, str(out))
    assert out.exists()
    assert path == str(out)
