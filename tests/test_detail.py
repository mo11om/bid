"""Tests for game-sequence detail capture (format_contract, dds_details, detail=)."""

from src.config import Config
from src.evaluation.metrics import dds_details, evaluate_dataset, format_contract
from src.harness.prompt_builder import ContextBuilder
from src.schema.dataset import BridgeBid
from src.data.mock_generator import records_from_labeled

LABELED = {
    "N": "S:AK7.H:QJ3.D:854.C:KT62",   # 13 HCP -> expert opens 1NT
    "E": "S:432.H:432.D:432.C:5432",
    "S": "S:T98.H:T98.D:T98.C:T987",
    "W": "S:765.H:765.D:765.C:6543",
}


def test_format_contract():
    assert format_contract(None) == "Passed out"
    assert format_contract((3, "NT", "N", 0)) == "3NT by N"
    assert format_contract((4, "S", "E", 1)) == "4S X by E"
    assert format_contract((6, "H", "S", 2)) == "6H XX by S"


class _ScriptedClient:
    """Returns a fixed call; rolls out by appending three passes."""

    def __init__(self, bid: str) -> None:
        self.builder = ContextBuilder()
        self._bid = bid

    def get_bid(self, prompt, history=None):
        return BridgeBid(thinking="stub", bid=self._bid)

    def rollout_auction(self, record, first_bid):
        return list(record.current_bidding) + [first_bid, "Pass", "Pass", "Pass"]


def test_dds_details_shape(monkeypatch):
    # Avoid the real endplay solver: stub the NS-score lookup deterministically.
    import src.evaluation.metrics as m

    monkeypatch.setattr(
        m, "ns_score_for_contract", lambda pbn, contract, vul: 400 if contract else 0
    )
    rec = records_from_labeled(LABELED, board_number=1)[0]  # North, expert 1NT
    detail = dds_details(rec, "2NT", "1NT", _ScriptedClient("2NT"), Config())

    assert detail["llm_auction"][0] == "2NT"
    assert detail["expert_auction"][0] == "1NT"
    assert detail["llm_contract"] == "2NT by N"
    assert detail["expert_contract"] == "1NT by N"
    assert detail["llm_score_ns"] == 400
    assert detail["imp_delta"] == 0
    assert detail["acceptable"] is True


def test_evaluate_dataset_detail_attaches_sequences(monkeypatch):
    import src.evaluation.metrics as m

    monkeypatch.setattr(
        m, "ns_score_for_contract", lambda pbn, contract, vul: 400 if contract else 0
    )
    records = records_from_labeled(LABELED, board_number=1)
    # Model always passes: matches the three Pass experts, differs from N's 1NT.
    summary = evaluate_dataset(records, _ScriptedClient("Pass"), detail=True)

    by_seat = {r["seat"]: r for r in summary["results"]}
    # Position context present on every result.
    assert by_seat["N"]["dealer"] == "N"
    assert by_seat["E"]["current_bidding"] == ["1NT"]
    # The mismatched seat (N) carries a full rollout detail block.
    assert "dds" in by_seat["N"]
    assert by_seat["N"]["dds"]["llm_auction"][0] == "Pass"
    assert by_seat["N"]["dds"]["expert_auction"][0] == "1NT"
    # Exact-match seats have no rollout block.
    assert "dds" not in by_seat["E"]


def test_detail_false_keeps_results_compact():
    records = records_from_labeled(LABELED, board_number=1)
    summary = evaluate_dataset(records, _ScriptedClient("Pass"), use_dds=False)
    assert "dealer" not in summary["results"][0]
    assert "dds" not in summary["results"][0]
