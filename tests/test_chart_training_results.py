import csv
from pathlib import Path
import tempfile
import unittest

from tools.chart_training_results import (
    DASHBOARD_SCRIPT,
    Series,
    _chart_ceiling,
    _derived_ratio,
    _evaluation_dashboard,
    _evaluation_set,
    _run_mode,
    _statistics,
    _tier_player_count_charts,
    build_dashboard,
    read_results,
)


class TrainingResultsChartTests(unittest.TestCase):
    def test_current_and_legacy_run_modes_and_derived_metrics(self):
        current = {"run_type": "evaluation", "run_mode": "evaluation_early"}
        legacy = {
            "run_type": "training",
            "training_exploration_mode": "zero_epsilon",
        }
        counters = {
            "move_action_count": "3",
            "spent_action_count": "12",
            "move_claim_conversions": "2",
            "moves_creating_claimable_route": "4",
            "sampled_training_decision_count": "1024",
            "trajectory_decision_count": "4096",
        }

        self.assertEqual(_run_mode(current), "evaluation_early")
        self.assertEqual(_evaluation_set(current), "early")
        self.assertEqual(_run_mode(legacy), "training_zero_epsilon")
        self.assertEqual(_derived_ratio(counters, "move_action_count", "spent_action_count"), 0.25)
        self.assertEqual(
            _derived_ratio(
                counters,
                "move_claim_conversions",
                "moves_creating_claimable_route",
            ),
            0.5,
        )
        self.assertEqual(
            _derived_ratio(
                counters,
                "sampled_training_decision_count",
                "trajectory_decision_count",
            ),
            0.25,
        )
        self.assertIsNone(_derived_ratio({}, "move_action_count", "spent_action_count"))

    def test_loss_statistics_use_whole_numbers_except_percentage_change(self):
        summary = _statistics(((1, 1000.25), (2, 2000.75)))

        self.assertIn("<span>1,500</span>", summary)
        self.assertNotIn("1,500.50", summary)
        self.assertIn("<span>100.0%</span>", summary)

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

    def test_training_loss_axis_ignores_evaluation_rows(self):
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
                        "latest_loss",
                        "rolling_mean_loss",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"game#": 1, "run_type": "training", "latest_loss": 10},
                        {
                            "game#": 2,
                            "batch#": 1,
                            "run_type": "evaluation",
                            "player_count": 3,
                            "latest_loss": 20,
                        },
                        {
                            "game#": 3,
                            "batch#": 1,
                            "run_type": "evaluation",
                            "player_count": 3,
                            "latest_loss": 30,
                        },
                        {"game#": 4, "run_type": "training", "latest_loss": 40},
                    )
                )

            _rows, series, _counts = read_results(path, 100)

            self.assertEqual(series["latest_loss", "training"].points, [(1.0, 10.0), (2.0, 40.0)])

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
            chart = _evaluation_dashboard(counts)

            self.assertIn("Evaluation — Standard", chart)
            self.assertIn("Evaluation loss by batch", DASHBOARD_SCRIPT)
            self.assertIn("Five-batch average", DASHBOARD_SCRIPT)
            self.assertIn("Win rate by tier", DASHBOARD_SCRIPT)
            self.assertIn("Average final score by tier", DASHBOARD_SCRIPT)
            self.assertIn("Average completed-game length", DASHBOARD_SCRIPT)
            self.assertIn("All maps", chart)
            self.assertIn("Map 1", chart)
            self.assertIn("All players", chart)
            self.assertIn("3 players", chart)
            self.assertIn('"players":"3"', chart)
            self.assertIn("Higher is better", DASHBOARD_SCRIPT)
            self.assertIn("Lower is generally better", DASHBOARD_SCRIPT)
            self.assertIn("Completed", DASHBOARD_SCRIPT)
            self.assertIn("Random-win baseline", DASHBOARD_SCRIPT)
            self.assertIn('class="svg-x-grid"', DASHBOARD_SCRIPT)
            self.assertIn("All latest evaluation games completed normally", DASHBOARD_SCRIPT)
            self.assertNotIn("Completion rate", chart)
            self.assertIn("data-evaluation-type", chart)
            self.assertEqual(chart.count("data-evaluation-panel"), 1)
            self.assertNotIn("data-map=", chart)

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

    def test_evaluation_movement_metrics_are_aggregated_and_charted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            fieldnames = (
                "game#",
                "batch#",
                "run_type",
                "map",
                "player_count",
                "action_count",
                "move_action_count",
                "spent_action_count",
                "pointless_move_workflows",
                "repeated_move_penalties",
                "all_move_turn_penalties",
                "moves_creating_claimable_route",
                "move_claim_conversions",
            )
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "game#": 1,
                            "batch#": 1,
                            "run_type": "evaluation",
                            "map": 2,
                            "player_count": 3,
                            "action_count": 400,
                            "move_action_count": 20,
                            "spent_action_count": 100,
                            "pointless_move_workflows": 2,
                            "repeated_move_penalties": 3,
                            "all_move_turn_penalties": 1,
                            "moves_creating_claimable_route": 4,
                            "move_claim_conversions": 2,
                        },
                        {
                            "game#": 2,
                            "batch#": 1,
                            "run_type": "evaluation",
                            "map": 2,
                            "player_count": 3,
                            "action_count": 500,
                            "move_action_count": 10,
                            "spent_action_count": 50,
                            "pointless_move_workflows": 1,
                            "repeated_move_penalties": 1,
                            "all_move_turn_penalties": 0,
                            "moves_creating_claimable_route": 2,
                            "move_claim_conversions": 1,
                        },
                    )
                )

            _rows, _series, counts = read_results(path, 100)
            chart = _evaluation_dashboard(counts)
            self.assertIn("Move % of paid actions", DASHBOARD_SCRIPT)
            self.assertIn("Movement pathology", DASHBOARD_SCRIPT)
            self.assertIn("Pointless Moves/game", DASHBOARD_SCRIPT)
            self.assertIn("Repeated-Move penalties/game", DASHBOARD_SCRIPT)
            self.assertIn("All-Move-turn penalties/game", DASHBOARD_SCRIPT)
            self.assertIn("Move → Claim conversion rate", DASHBOARD_SCRIPT)
            self.assertNotIn("Pointless Move workflows per game", chart)
            self.assertEqual(chart.count("data-evaluation-panel"), 1)
            self.assertIn('"move_action_count":30.0', chart)
            self.assertIn('"pointless_move_workflows":3.0', chart)
            self.assertIn('"map":"2","players":"3"', chart)

    def test_evaluation_types_share_one_filterable_dashboard_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            fieldnames = (
                "game#",
                "batch#",
                "run_type",
                "run_mode",
                "evaluation_suite_version",
                "evaluation_suite_size",
                "map",
                "player_count",
                "winner_tier",
                "tier_to_seat_assignments",
                "final_player_scores",
                "completion_reason",
                "action_count",
                "move_action_count",
                "spent_action_count",
                "pointless_move_workflows",
                "repeated_move_penalties",
                "all_move_turn_penalties",
                "moves_creating_claimable_route",
                "move_claim_conversions",
            )
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "game#": 1,
                            "batch#": 2,
                            "run_type": "evaluation",
                            "run_mode": "evaluation_mid_late_end",
                            "evaluation_suite_version": 5,
                            "evaluation_suite_size": 1,
                            "map": 1,
                            "player_count": 3,
                            "winner_tier": "[3]",
                            "tier_to_seat_assignments": "[1, 3, 5]",
                            "final_player_scores": "[20, 30, 21]",
                            "completion_reason": "20_points",
                            "action_count": 100,
                        },
                        {
                            "game#": 2,
                            "batch#": 2,
                            "run_type": "evaluation",
                            "run_mode": "evaluation_early",
                            "evaluation_suite_version": 5,
                            "evaluation_suite_size": 1,
                            "map": 2,
                            "player_count": 5,
                            "winner_tier": "[1]",
                            "tier_to_seat_assignments": "[1, 2, 3, 4, 5]",
                            "final_player_scores": "[30, 20, 19, 18, 17]",
                            "completion_reason": "action_limit",
                            "action_count": 10000,
                            "move_action_count": 20,
                            "spent_action_count": 100,
                            "pointless_move_workflows": 2,
                            "repeated_move_penalties": 1,
                            "all_move_turn_penalties": 1,
                            "moves_creating_claimable_route": 4,
                            "move_claim_conversions": 2,
                        },
                    )
                )

            _rows, _series, counts = read_results(path, 100)
            dashboard = _evaluation_dashboard(counts)

            self.assertIn("Evaluation — Standard", dashboard)
            self.assertEqual(dashboard.count('class="card evaluation-performance"'), 1)
            self.assertEqual(dashboard.count("data-evaluation-panel"), 1)
            self.assertEqual(dashboard.count("data-evaluation-data"), 1)
            self.assertIn('<option value="standard">Standard</option>', dashboard)
            self.assertIn('<option value="early">Early</option>', dashboard)
            self.assertNotIn("Mixed Development", dashboard)
            self.assertIn("data-evaluation-map", dashboard)
            self.assertIn("data-evaluation-players", dashboard)
            self.assertIn("datasets[mode]", DASHBOARD_SCRIPT)
            self.assertIn("data-evaluation-title", DASHBOARD_SCRIPT)
            self.assertIn("Tier 1 win rate by player count", DASHBOARD_SCRIPT)
            self.assertIn("Average interactions per early-game evaluation", DASHBOARD_SCRIPT)
            self.assertIn("Early-game timeout rate", DASHBOARD_SCRIPT)
            self.assertIn("Move % of paid actions", DASHBOARD_SCRIPT)
            self.assertIn("Move → Claim conversion rate", DASHBOARD_SCRIPT)
            self.assertIn('"standard":', dashboard)
            self.assertIn('"early":', dashboard)
            self.assertIn('"map":"2","players":"5"', dashboard)

            self.assertIn("Move → Claim conversion rate", DASHBOARD_SCRIPT)

    def test_dashboard_uses_one_loss_chart_and_compact_game_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("game#", "run_type", "completion_reason", "latest_loss"),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "game#": 1,
                            "run_type": "training",
                            "completion_reason": "20_points",
                            "latest_loss": 100,
                        },
                        {
                            "game#": 2,
                            "run_type": "training",
                            "completion_reason": "action_limit",
                            "latest_loss": 200,
                        },
                    )
                )

            rows, series, counts = read_results(path, 100)
            dashboard = build_dashboard(rows, series, counts, path)

            self.assertEqual(dashboard.count("Latest training loss"), 2)
            self.assertNotIn("Rolling mean loss", dashboard)
            self.assertIn("Training games</strong><span>2", dashboard)
            self.assertIn("Evaluation games</strong><span>0", dashboard)
            self.assertIn("Timeouts</strong><span>1", dashboard)
            self.assertNotIn("Completion results", dashboard)
            self.assertNotIn("Game types", dashboard)


if __name__ == "__main__":
    unittest.main()
