"""Verify the redeal interop in mock_generator with a fake redeal module.

Confirms ``Deal.prepare()`` is treated as a dealer factory, that per-suit
holdings are read via ``str(hand.<suit>)``, and that records serialize to JSONL
and round-trip back into MockDealRecord.
"""

import sys
import types

import pytest

from src.schema.dataset import MockDealRecord


class _Holding(str):
    """Stand-in for a redeal Holding; str() yields the ranks."""


class _Hand:
    def __init__(self, s, h, d, c):
        self.spades = _Holding(s)
        self.hearts = _Holding(h)
        self.diamonds = _Holding(d)
        self.clubs = _Holding(c)


class _Deal:
    def __init__(self, n, e, s, w):
        self.north, self.east, self.south, self.west = n, e, s, w


def _install_fake_redeal():
    deal = _Deal(
        _Hand("AK7", "QJ3", "854", "KT62"),
        _Hand("QJ2", "A8", "AKQ", "AQJ9"),
        _Hand("T98", "KT9", "JT9", "8753"),
        _Hand("6543", "7654", "762", "4"),
    )

    class Deal:
        @classmethod
        def prepare(cls):
            return lambda: deal

    redeal = types.ModuleType("redeal")
    redeal.Deal = Deal
    sys.modules["redeal"] = redeal


@pytest.fixture
def cleanup_redeal():
    yield
    sys.modules.pop("redeal", None)


def test_generate_mock_dataset_roundtrip(tmp_path, cleanup_redeal):
    _install_fake_redeal()
    from src.data.mock_generator import generate_mock_dataset

    out = tmp_path / "ds.jsonl"
    written = generate_mock_dataset(2, str(out))
    assert written == 8  # 2 deals x 4 seats

    records = [
        MockDealRecord.model_validate_json(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 8

    north_b1 = next(r for r in records if r.id == "b1-N")
    assert north_b1.hand == "S:AK7.H:QJ3.D:854.C:KT62"
    assert north_b1.hcp == 13
    assert north_b1.deal_pbn.startswith("N:AK7.QJ3.854.KT62 ")
    assert north_b1.all_hands["E"] == "QJ2.A8.AKQ.AQJ9"
