from __future__ import annotations

import unittest

from evaluation.metrics import (
    aggregate_metrics,
    efficiency,
    hit_rate_at_k,
    mean_reciprocal_rank,
    mean_turn_to_conversion,
    technical_score,
)


class MetricsTests(unittest.TestCase):
    def test_official_metric_formulas(self) -> None:
        ranked = [["a", "target"], ["target"], ["x"]]
        targets = ["target", "target", "target"]
        self.assertAlmostEqual(hit_rate_at_k(ranked, targets, 10), 2 / 3)
        self.assertAlmostEqual(mean_reciprocal_rank(ranked, targets, 10), (0.5 + 1.0) / 3)
        self.assertAlmostEqual(mean_turn_to_conversion([1, 2, None]), 14 / 3)
        self.assertAlmostEqual(efficiency(14 / 3), (11 - 14 / 3) / 10)
        self.assertAlmostEqual(technical_score(2 / 3, 0.5, 0.6333333333), 0.61)

    def test_aggregate_scenario_metrics(self) -> None:
        result = aggregate_metrics(
            [
                {"scenario_type": "buying", "ranked_ids": ["a", "t"], "target_id": "t", "first_hit_turn": 1},
                {"scenario_type": "browsing", "ranked_ids": ["t"], "target_id": "t", "first_hit_turn": 2},
            ]
        )
        self.assertEqual(result["sample_count"], 2)
        self.assertIn("buying", result["scenario_metrics"])
        self.assertEqual(result["scenario_metrics"]["browsing"]["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
