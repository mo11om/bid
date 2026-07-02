"""Metrics tests that do not require endplay (scoring, settling, IMPs)."""

import pytest

from src.evaluation.metrics import (
    calculate_accuracy,
    contract_score,
    imp_diff,
    settle_contract,
)


def test_calculate_accuracy():
    assert calculate_accuracy(["Pass", "1NT"], ["Pass", "1NT"]) == 1.0
    assert calculate_accuracy(["Pass", "1NT"], ["Pass", "2C"]) == 0.5
    assert calculate_accuracy(["p"], ["Pass"]) == 1.0  # normalization
    assert calculate_accuracy([], []) == 0.0


def test_contract_score_made_game():
    # 3NT making exactly (9 tricks), not vulnerable: 100 trick + 300 game = 400.
    assert contract_score(3, "NT", 9, vul=False) == 400
    # 4S making (10 tricks) NV: 120 + 300 = 420.
    assert contract_score(4, "S", 10, vul=False) == 420
    # 4S vulnerable: 120 + 500 = 620.
    assert contract_score(4, "S", 10, vul=True) == 620


def test_contract_score_partscore_and_down():
    # 2H making (8 tricks) NV: 60 + 50 = 110.
    assert contract_score(2, "H", 8, vul=False) == 110
    # 3NT down 2 NV undoubled: -100.
    assert contract_score(3, "NT", 7, vul=False) == -100
    # 1S X down 1 vulnerable: -200.
    assert contract_score(1, "S", 6, vul=True, doubled=1) == -200


def test_imp_diff():
    assert imp_diff(420, 420) == 0
    assert imp_diff(620, 420) == 5   # 200 diff -> 5 IMPs
    assert imp_diff(420, 620) == -5
    assert imp_diff(400, 420) == -1  # 20 diff -> 1 IMP (still within 1-IMP rule)
    assert imp_diff(420, 425) == 0   # <20 diff -> 0 IMPs


def test_settle_contract_basic():
    # Dealer N: N opens 1NT, all pass. Declarer = N.
    c = settle_contract(["1NT", "Pass", "Pass", "Pass"], dealer="N")
    assert c == (1, "NT", "N", 0)


def test_settle_contract_declarer_resolution():
    # Dealer N. N:1H Pass S:2H all pass. Both N and S bid hearts; first is N.
    auction = ["1H", "Pass", "2H", "Pass", "Pass", "Pass"]
    c = settle_contract(auction, dealer="N")
    assert c == (2, "H", "N", 0)


def test_settle_contract_passed_out():
    assert settle_contract(["Pass", "Pass", "Pass", "Pass"], dealer="N") is None


def test_settle_contract_doubled():
    # Dealer E. E:1S, S:X, all pass. Declarer = E, doubled.
    auction = ["1S", "X", "Pass", "Pass", "Pass"]
    c = settle_contract(auction, dealer="E")
    assert c == (1, "S", "E", 1)
