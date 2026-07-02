"""Scenario-based simulation (Phase 4).

Builds targeted deals with ``redeal`` (e.g. force N/S into a chosen HCP band to
probe partscore vs game competitive bidding), feeds them through
``LocalLLMClient``, and evaluates them with the Phase 3 metrics — a complete
end-to-end smoke test of the pipeline.
"""

from __future__ import annotations

from typing import List, Tuple

from src.config import Config, DEFAULT_CONFIG
from src.data.mock_generator import records_for_deal
from src.evaluation.metrics import evaluate_dataset
from src.harness.llm_client import LocalLLMClient
from src.schema.dataset import MockDealRecord

HCPRange = Tuple[int, int]

# Rejection-sampling safety factor: give up after this many tries per wanted deal.
_MAX_ATTEMPT_FACTOR = 500


class ScenarioRunner:
    """Generates constrained scenarios and runs them through the eval pipeline."""

    def __init__(
        self, client: LocalLLMClient = None, config: Config = DEFAULT_CONFIG
    ) -> None:
        self.config = config
        self.client = client or LocalLLMClient(config)

    # ------------------------------------------------------------------ #
    def build_scenario(
        self,
        count: int,
        north_hcp: HCPRange = (0, 40),
        south_hcp: HCPRange = (0, 40),
        start_board: int = 1,
    ) -> List[MockDealRecord]:
        """Generate ``count`` deals with N/S inside the given HCP bands.

        Returns the flattened per-seat records (4 per deal).
        """
        from redeal import Deal

        dealer = Deal.prepare()
        records: List[MockDealRecord] = []
        produced = 0
        attempts = 0
        max_attempts = max(count * _MAX_ATTEMPT_FACTOR, _MAX_ATTEMPT_FACTOR)
        board = start_board

        while produced < count and attempts < max_attempts:
            deal = dealer()
            attempts += 1
            if (
                north_hcp[0] <= deal.north.hcp <= north_hcp[1]
                and south_hcp[0] <= deal.south.hcp <= south_hcp[1]
            ):
                records.extend(records_for_deal(deal, board))
                board += 1
                produced += 1

        if produced < count:
            raise RuntimeError(
                f"Only generated {produced}/{count} deals matching constraints "
                f"after {attempts} attempts; loosen the HCP ranges."
            )
        return records

    # ------------------------------------------------------------------ #
    def run(
        self,
        count: int,
        north_hcp: HCPRange = (0, 40),
        south_hcp: HCPRange = (0, 40),
        detail: bool = False,
    ) -> dict:
        """Build a scenario and evaluate it end-to-end."""
        records = self.build_scenario(count, north_hcp=north_hcp, south_hcp=south_hcp)
        return evaluate_dataset(records, self.client, self.config, detail=detail)
