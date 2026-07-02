"""Pydantic models for datasets and structured LLM output.

``MockDealRecord`` extends the spec's fields with the *full* deal so the
double-dummy evaluator and the self-play auction rollout have complete
information. The prompt layer (``ContextBuilder``) deliberately reads ONLY the
masked subset (``hand``, ``hcp``, ``current_bidding``) so the model never sees
the other hands.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


SEAT_PATTERN = ("N", "E", "S", "W")
VUL_VALUES = ("None", "NS", "EW", "Both")


class BridgeBid(BaseModel):
    """Structured output the LLM is asked to produce for a single call."""

    thinking: str = Field(default="", description="Brief reasoning for the call.")
    bid: str = Field(description="The call, e.g. 'Pass', '1NT', 'X', 'XX'.")


class MockDealRecord(BaseModel):
    """One evaluation position: a single seat to act on a known deal.

    Masked fields (shown to the model): ``hand``, ``hcp``, ``current_bidding``.
    Full-information fields (used only by the evaluator / rollout, never shown):
    ``deal_pbn``, ``all_hands``, ``dealer``, ``vulnerability``.
    """

    id: str
    seat: str = Field(description="Active seat: N/E/S/W.")
    hand: str = Field(description="Active seat's hand, e.g. 'S:AK7.H:QJ3.D:854.C:KT62'.")
    hcp: int = Field(ge=0, le=40)
    shape: str = Field(description="Distribution, e.g. '4-3-3-3'.")
    ltc: int = Field(ge=0, le=12, description="Standard Losing Trick Count.")
    current_bidding: List[str] = Field(
        default_factory=list,
        description="Auction so far, from dealer up to (not including) this seat.",
    )
    expert_bid: str = Field(description="Reference call to compare against.")

    # --- full-information fields (never placed in the prompt) ---
    deal_pbn: str = Field(
        description="Full 4-hand PBN, e.g. 'N:AK7.QJ3.854.KT62  E:... S:... W:...'."
    )
    all_hands: Dict[str, str] = Field(
        default_factory=dict,
        description="Seat -> hand string for all four seats (plain PBN holdings).",
    )
    dealer: str = Field(default="N", description="Dealer seat: N/E/S/W.")
    vulnerability: str = Field(default="None", description="One of None/NS/EW/Both.")

    def masked_view(self) -> Dict[str, object]:
        """Exactly the fields the model is allowed to see."""
        return {
            "seat": self.seat,
            "hand": self.hand,
            "hcp": self.hcp,
            "current_bidding": list(self.current_bidding),
        }
