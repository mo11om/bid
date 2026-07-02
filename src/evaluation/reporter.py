"""Reporting (Phase 3): accuracy vs HCP visualization.

``generate_report`` takes per-position results and produces a grouped bar chart
comparing *exact-match* accuracy and *DDS-acceptable* accuracy across the HCP
buckets 0-10, 11-15, 16+. Uses a non-interactive matplotlib backend so it works
headless.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.bridge import HCP_BUCKETS, hcp_bucket  # noqa: E402


def aggregate_by_bucket(eval_results: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    """Aggregate accuracy per HCP bucket.

    Each result dict must contain ``hcp`` (int), ``exact_match`` (bool) and
    ``dds_acceptable`` (bool).
    """
    counts = {b: {"n": 0, "exact": 0, "dds": 0} for b in HCP_BUCKETS}
    for r in eval_results:
        b = hcp_bucket(int(r["hcp"]))
        counts[b]["n"] += 1
        counts[b]["exact"] += 1 if r.get("exact_match") else 0
        counts[b]["dds"] += 1 if r.get("dds_acceptable") else 0

    summary: Dict[str, Dict[str, float]] = {}
    for b in HCP_BUCKETS:
        n = counts[b]["n"]
        summary[b] = {
            "n": n,
            "exact_accuracy": round(counts[b]["exact"] / n, 4) if n else 0.0,
            "dds_accuracy": round(counts[b]["dds"] / n, 4) if n else 0.0,
        }
    return summary


def generate_report(eval_results: Sequence[Dict], output_path: str) -> str:
    """Render the accuracy-vs-HCP bar chart to ``output_path`` (PNG). Returns the path."""
    summary = aggregate_by_bucket(eval_results)
    buckets: List[str] = HCP_BUCKETS
    exact = [summary[b]["exact_accuracy"] for b in buckets]
    dds = [summary[b]["dds_accuracy"] for b in buckets]
    ns = [int(summary[b]["n"]) for b in buckets]

    x = range(len(buckets))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], exact, width, label="Exact match")
    ax.bar([i + width / 2 for i in x], dds, width, label="DDS-acceptable")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(buckets, ns)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("HCP range")
    ax.set_title("Bidding accuracy by HCP range")
    ax.legend()

    for i, (e, d) in enumerate(zip(exact, dds)):
        ax.text(i - width / 2, e + 0.02, f"{e:.0%}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, d + 0.02, f"{d:.0%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path
