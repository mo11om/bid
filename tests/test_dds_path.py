"""Verify the DDS evaluation composition with a fake endplay solver.

endplay ships a C double-dummy solver that isn't available in every
environment, so we inject a fake ``endplay`` exposing the same surface
(``Deal``, ``Denom``, ``Player``, ``calc_dd_table`` with ``table[denom, player]``
indexing) to exercise settle -> trick-lookup -> score -> IMP-threshold wiring.
"""

import sys
import types

import pytest

from src.config import Config
from src.schema.dataset import MockDealRecord


# --------------------------------------------------------------------------- #
def _install_fake_endplay(tricks_by_strain):
    """Register a fake endplay where N takes ``tricks_by_strain[strain]`` tricks."""

    class _Sym:
        def __init__(self, name):
            self.name = name

        def __repr__(self):
            return self.name

    denom = types.SimpleNamespace(
        clubs=_Sym("C"), diamonds=_Sym("D"), hearts=_Sym("H"),
        spades=_Sym("S"), nt=_Sym("NT"),
    )
    player = types.SimpleNamespace(
        north=_Sym("N"), east=_Sym("E"), south=_Sym("S"), west=_Sym("W"),
    )

    strain_lookup = {
        denom.clubs: "C", denom.diamonds: "D", denom.hearts: "H",
        denom.spades: "S", denom.nt: "NT",
    }

    class _Table:
        def __getitem__(self, key):
            d, p = key
            # Only N declares in these tests.
            return tricks_by_strain.get(strain_lookup[d], 0)

    class _Deal:
        def __init__(self, pbn):
            self.pbn = pbn

    endplay = types.ModuleType("endplay")
    dds = types.ModuleType("endplay.dds")
    typesmod = types.ModuleType("endplay.types")
    dds.calc_dd_table = lambda deal: _Table()
    typesmod.Deal = _Deal
    typesmod.Denom = denom
    typesmod.Player = player
    endplay.dds = dds
    endplay.types = typesmod
    sys.modules["endplay"] = endplay
    sys.modules["endplay.dds"] = dds
    sys.modules["endplay.types"] = typesmod


@pytest.fixture
def cleanup_endplay():
    yield
    for m in ("endplay", "endplay.dds", "endplay.types"):
        sys.modules.pop(m, None)


class StubRolloutClient:
    """Closes any auction by appending three passes after the first bid."""

    def rollout_auction(self, record, first_bid):
        return [first_bid, "Pass", "Pass", "Pass"]


def _record():
    return MockDealRecord(
        id="d1", seat="N", hand="S:AKQJ.H:AK.D:AKQ.C:AKQ2", hcp=28,
        shape="4-4-3-2", ltc=2, current_bidding=[], expert_bid="3NT",
        deal_pbn="N:AKQJ.AK.AKQ.AKQ2 E:... S:... W:...",
        all_hands={"N": "x", "E": "x", "S": "x", "W": "x"},
        dealer="N", vulnerability="None",
    )


def test_dds_accepts_within_one_imp(cleanup_endplay):
    # N makes 4S=10 tricks (420) vs 3NT=9 tricks (400): 20 pts -> 1 IMP -> accept.
    _install_fake_endplay({"S": 10, "NT": 9})
    from src.evaluation.metrics import evaluate_with_dds

    ok = evaluate_with_dds(
        _record(), "4S", "3NT", StubRolloutClient(), Config(threshold_mode="imp", threshold_n=1)
    )
    assert ok is True


def test_dds_rejects_large_gap(cleanup_endplay):
    # N takes only 7 tricks in spades: 4S down 3 (-150) vs 3NT 400: ~11 IMPs -> reject.
    _install_fake_endplay({"S": 7, "NT": 9})
    from src.evaluation.metrics import evaluate_with_dds

    ok = evaluate_with_dds(
        _record(), "4S", "3NT", StubRolloutClient(), Config(threshold_mode="imp", threshold_n=1)
    )
    assert ok is False


def test_dds_exact_match_short_circuits(cleanup_endplay):
    # Same bid as expert -> accepted without touching the solver.
    from src.evaluation.metrics import evaluate_with_dds

    class Boom:
        def rollout_auction(self, *a):
            raise AssertionError("should not roll out on exact match")

    assert evaluate_with_dds(_record(), "3NT", "3NT", Boom(), Config()) is True


# --------------------------------------------------------------------------- #
# Asymmetric acceptability (dds_rule)
# --------------------------------------------------------------------------- #
def test_asymmetric_accepts_model_strictly_better_ns(cleanup_endplay):
    # N seat: 4S+3 = 510 vs 3NT = 400 -> ~3 IMPs in the model's favour.
    _install_fake_endplay({"S": 13, "NT": 9})
    from src.evaluation.metrics import evaluate_with_dds

    rec = _record()  # seat N
    assert evaluate_with_dds(
        rec, "4S", "3NT", StubRolloutClient(),
        Config(dds_rule="asymmetric", threshold_n=1),
    ) is True
    assert evaluate_with_dds(
        rec, "4S", "3NT", StubRolloutClient(),
        Config(dds_rule="symmetric", threshold_n=1),
    ) is False


def test_asymmetric_orients_by_seat_for_ew(cleanup_endplay):
    # Both rollouts still declared by N (NS side). The model's line scores
    # LESS for NS (4S down 3 = -150 vs 3NT = 400), i.e. BETTER for an E/W
    # bidder: asymmetric must accept for seat E, reject for seat N.
    _install_fake_endplay({"S": 7, "NT": 9})
    from src.evaluation.metrics import dds_details

    rec_e = _record()
    rec_e.seat = "E"
    d = dds_details(rec_e, "4S", "3NT", StubRolloutClient(),
                    Config(dds_rule="asymmetric", threshold_n=1))
    assert d["model_gain_imp"] > 0
    assert d["acceptable"] is True

    rec_n = _record()  # seat N: same delta is a loss for the bidder
    d = dds_details(rec_n, "4S", "3NT", StubRolloutClient(),
                    Config(dds_rule="asymmetric", threshold_n=1))
    assert d["model_gain_imp"] < 0
    assert d["acceptable"] is False


def test_dds_rule_validated():
    with pytest.raises(ValueError):
        Config(dds_rule="lenient")
