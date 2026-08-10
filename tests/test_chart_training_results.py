import csv
from pathlib import Path
import tempfile
import unittest

from tools.chart_training_results import (
    DASHBOARD_SCRIPT,
    Series,
    _chart_ceiling,
    _evaluation_chart,
    _tier_player_count_charts,
    read_results,
)


class TrainingResultsChartTests(unittest.TestCase):
    def test_loss_charts_keep_history_and_group_only_the_visible_window(self):
        series = Series(max_points=10)
        for game_number in range(1, 26):
            series.add(game_number, game_number * 2)

        self.assertEqual(len(series.points), 25)
        self.assertIn("const MAX_VISIBLE_GROUPS = 750", DASHBOARD_SCRIPT)
        self.assertIn(
            "const groupSize = Math.ceil(points.length / MAX_VISIBLE_GROUPS)", DASHBOARD_SCRIPT
        )
        self.assertIn("minimum: Math.min(...values)", DASHBOARD_SCRIPT)
        self.assertIn("maximum: Math.max(...values)", DASHBOARD_SCRIPT)
        self.assertIn("data.median.filter", DASHBOARD_SCRIPT)

    def test_chart_ceiling_adds_rounded_headroom(self):
        self.assertEqual(_chart_ceiling(31, 10, minimum=40, maximum=100), 40)
        self.assertEqual(_chart_ceiling(54, 10, minimum=40, maximum=100), 60)
        self.assertEqual(_chart_ceiling(100, 10, minimum=40, maximum=100), 100)
        self.assertEqual(_chart_ceiling(40, 5, minimum=5), 45)

    def test_evaluation_rows_are_grouped_into_batch_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "game#",
                        "batch#",
                        "run_type",
                        "player_count",
                        "winner_tier",
                        "tier_to_seat_assignments",
                        "final_player_scores",
                        "latest_loss",
                        "evaluation_suite_size",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "game#": 1,
                            "batch#": 4,
                            "run_type": "evaluation",
                            "player_count": 3,
                            "winner_tier": "[1]",
                            "tier_to_seat_assignments": "[1, 3, 5]",
                            "final_player_scores": "[40, 30, 20]",
                            "latest_loss": "800",
                            "evaluation_suite_size": "2",
                        },
                        {
                            "game#": 2,
                            "batch#": 4,
                            "run_type": "evaluation",
                            "player_count": 5,
                            "winner_tier": "[3]",
                            "tier_to_seat_assignments": "[1, 2, 3, 4, 5]",
                            "final_player_scores": "[30, 31, 40, 29, 28]",
                            "latest_loss": "1000",
                            "evaluation_suite_size": "2",
                        },
                    )
                )

            _rows, _series, counts = read_results(path, 100)
            chart = _evaluation_chart(
                counts["evaluation_batches"],
                counts["evaluation_map_batches"],
                counts["evaluation_player_batches"],
                counts["evaluation_map_player_batches"],
            )

            self.assertIn("Evaluation results", chart)
            self.assertIn("Evaluation loss by batch", chart)
            self.assertIn("Five-batch average", chart)
            self.assertIn("2/2 boards completed", chart)
            self.assertIn("Win rate by tier", chart)
            self.assertIn("Average final score by tier", chart)
            self.assertIn("Average completed-game length", chart)
            self.assertIn("All maps", chart)
            self.assertIn("Map 1", chart)
            self.assertIn("All players", chart)
            self.assertIn("3 players", chart)
            self.assertIn('data-players="3"', chart)
            self.assertIn("Higher is better", chart)
            self.assertIn("Lower is generally better", chart)
            self.assertIn("Completed", chart)
            self.assertIn("Random-win baseline", chart)
            self.assertIn('class="svg-x-grid"', chart)
            self.assertIn("All latest evaluation games completed normally", chart)
            self.assertNotIn("Completion rate", chart)
            self.assertNotIn("data-evaluation-select", chart)

            tier_chart = _tier_player_count_charts(counts)
            self.assertIn("Tier performance by player count", tier_chart)
            self.assertIn("T1", tier_chart)
            self.assertIn("3p", tier_chart)
            self.assertIn("4p", tier_chart)
            self.assertIn("5p", tier_chart)
            self.assertNotIn("N/A", tier_chart)
            self.assertIn("33.3%, 25%, or 20%", tier_chart)
            self.assertIn("Only tiers assigned at that player count are shown", tier_chart)
            win_chart = tier_chart.split("Average final score", 1)[0]
            self.assertNotIn("N/A", win_chart)
            self.assertEqual(
                tier_chart.count(
                    "Each result shows the average score, the lowest-to-highest range"
                ),
                1,
            )
            self.assertNotIn("data-player-count-select", tier_chart)


if __name__ == "__main__":
    unittest.main()
