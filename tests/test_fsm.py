"""Legality guardrail tests."""

from src.harness.fsm_guardrail import BiddingFSM

fsm = BiddingFSM()


def test_pass_always_legal():
    assert fsm.is_valid_bid([], "Pass")
    assert fsm.is_valid_bid(["1NT", "Pass"], "Pass")


def test_contract_must_outrank():
    assert fsm.is_valid_bid(["1NT"], "2C")
    assert not fsm.is_valid_bid(["1NT"], "1S")  # 1S < 1NT
    assert not fsm.is_valid_bid(["2C"], "2C")  # equal not allowed
    assert fsm.is_valid_bid(["1C", "Pass", "1H"], "1S")


def test_opening_any_contract_legal():
    assert fsm.is_valid_bid([], "1C")
    assert fsm.is_valid_bid(["Pass", "Pass"], "1NT")


def test_double_legality():
    # Opponent (seat to act = index 1) doubling opener's 1NT (seat 0).
    assert fsm.is_valid_bid(["1NT"], "X")
    # Cannot double partner's bid (seat 2 over seat 0).
    assert not fsm.is_valid_bid(["1NT", "Pass"], "X")
    # Cannot double when nothing bid.
    assert not fsm.is_valid_bid([], "X")
    # Cannot double a pass.
    assert not fsm.is_valid_bid(["Pass"], "X")


def test_redouble_legality():
    # 1NT (seat0) - X (seat1) - XX (seat2, partner of opener) is legal.
    assert fsm.is_valid_bid(["1NT", "X"], "XX")
    # XX over a plain contract bid is illegal.
    assert not fsm.is_valid_bid(["1NT"], "XX")


def test_garbage_rejected():
    assert not fsm.is_valid_bid([], "banana")
    assert not fsm.is_valid_bid([], "8NT")
