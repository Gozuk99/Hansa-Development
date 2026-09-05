import csv
from pathlib import Path
import tempfile
import unittest

from tools.chart_training_results import (
    DASHBOARD_SCRIPT,
    NORMAL_MOVE_CAPACITY_FIELDS,
    Series,
    _chart_ceiling,
    _derived_ratio,
    _evaluation_dashboard,
    _evaluation_set,
    _run,
    _statistics,
    _tier_player_count_charts,
    build_dashboard,
    read_results,
)


class TrainingResultsChartTests(unittest.TestCase):
    def test_current_and_legacy_run_modes_and_derived_metrics(self):
        current = {"run": "evaluation_early"}
        legacy = {
            "run_type": "training",
            "training_exploration_mode": "zero_epsilon",
        }
        counters = {
            "move_action_count": "3",
            "spent_action_count": "12",
            "move_claim_conversions": "2",
            "moves_creating_claimable_route": "4",
            "sampled_training_decisions": "1024",
            "total_training_decisions": "4096",
        }

        self.assertEqual(_run(current), "evaluation_early")
        self.assertEqual(_evaluation_set(current), "early")
        self.assertEqual(_run(legacy), "training_end_zero_epsilon")
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
                "sampled_training_decisions",
                "total_training_decisions",
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
            self.assertIn("Average interactions per game", DASHBOARD_SCRIPT)
            self.assertIn("Average paid actions per game", DASHBOARD_SCRIPT)
            self.assertIn("All maps", chart)
            self.assertIn("Map 1", chart)
            self.assertIn("All players", chart)
            self.assertIn("3 players", chart)
            self.assertIn('"players":"3"', chart)
            self.assertIn("Higher is better", DASHBOARD_SCRIPT)
            self.assertIn("Counts every neural-network decision", DASHBOARD_SCRIPT)
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

    def test_evaluation_win_chart_handles_tiers_with_zero_wins(self):
        self.assertIn(
            "(entry.tierWins[tier] || 0) / entry.tierGames[tier] * 100",
            DASHBOARD_SCRIPT,
        )
        self.assertIn(
            "series.flatMap(item => item.values).filter(Number.isFinite)",
            DASHBOARD_SCRIPT,
        )
        self.assertIn(
            ".filter(Number.isFinite);",
            DASHBOARD_SCRIPT,
        )

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
                "pointless_normal_move_workflows",
                "pointless_move_any2_workflows",
                "immediate_one_piece_q_undos",
                "repeated_move_penalties",
                "all_move_turn_penalties",
                "moves_creating_claimable_route",
                "move_claim_conversions",
                "single_piece_moves_creating_claimable_route",
                "single_piece_move_claim_conversions",
                "move_claim_reward_awarded",
                "move_claim_reward_blocked_already_claimable",
                "consecutive_move1_pairs",
                "consecutive_move1_pairs_with_multiple_available",
                "consecutive_move1_pairs_move2_capacity",
                "consecutive_move1_pairs_move3_capacity",
                "consecutive_move1_pairs_move4_capacity",
                "consecutive_move1_pairs_move5_capacity",
                "avoidable_extra_move_actions",
                "single_piece_moves",
                "full_effective_capacity_moves",
                "under_effective_capacity_moves",
                *NORMAL_MOVE_CAPACITY_FIELDS,
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
                            "pointless_move_workflows": 5,
                            "pointless_normal_move_workflows": 2,
                            "pointless_move_any2_workflows": 3,
                            "immediate_one_piece_q_undos": 2,
                            "repeated_move_penalties": 3,
                            "all_move_turn_penalties": 1,
                            "moves_creating_claimable_route": 4,
                            "move_claim_conversions": 2,
                            "single_piece_moves_creating_claimable_route": 3,
                            "single_piece_move_claim_conversions": 1,
                            "move_claim_reward_awarded": 2,
                            "move_claim_reward_blocked_already_claimable": 1,
                            "consecutive_move1_pairs": 5,
                            "consecutive_move1_pairs_with_multiple_available": 4,
                            "consecutive_move1_pairs_move2_capacity": 1,
                            "consecutive_move1_pairs_move3_capacity": 1,
                            "consecutive_move1_pairs_move4_capacity": 2,
                            "consecutive_move1_pairs_move5_capacity": 1,
                            "avoidable_extra_move_actions": 3,
                            "single_piece_moves": 12,
                            "full_effective_capacity_moves": 13,
                            "under_effective_capacity_moves": 7,
                            "normal_move_effective_capacity_1_moves": 8,
                            "normal_move_effective_capacity_2_moves": 4,
                            "normal_move_effective_capacity_3_moves": 3,
                            "normal_move_effective_capacity_4_moves": 3,
                            "normal_move_effective_capacity_5_moves": 2,
                            "normal_move_capacity_1_moved_1": 8,
                            "normal_move_capacity_2_moved_1": 2,
                            "normal_move_capacity_2_moved_2": 2,
                            "normal_move_capacity_3_moved_1": 1,
                            "normal_move_capacity_3_moved_2": 1,
                            "normal_move_capacity_3_moved_3": 1,
                            "normal_move_capacity_4_moved_1": 0,
                            "normal_move_capacity_4_moved_2": 1,
                            "normal_move_capacity_4_moved_3": 0,
                            "normal_move_capacity_4_moved_4": 2,
                            "normal_move_capacity_5_moved_1": 1,
                            "normal_move_capacity_5_moved_2": 0,
                            "normal_move_capacity_5_moved_3": 1,
                            "normal_move_capacity_5_moved_4": 0,
                            "normal_move_capacity_5_moved_5": 0,
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
                            "pointless_normal_move_workflows": 1,
                            "pointless_move_any2_workflows": 0,
                            "immediate_one_piece_q_undos": 1,
                            "repeated_move_penalties": 1,
                            "all_move_turn_penalties": 0,
                            "moves_creating_claimable_route": 2,
                            "move_claim_conversions": 1,
                            "single_piece_moves_creating_claimable_route": 1,
                            "single_piece_move_claim_conversions": 1,
                            "move_claim_reward_awarded": 1,
                            "move_claim_reward_blocked_already_claimable": 2,
                            "consecutive_move1_pairs": 2,
                            "consecutive_move1_pairs_with_multiple_available": 1,
                            "consecutive_move1_pairs_move2_capacity": 0,
                            "consecutive_move1_pairs_move3_capacity": 1,
                            "consecutive_move1_pairs_move4_capacity": 1,
                            "consecutive_move1_pairs_move5_capacity": 0,
                            "avoidable_extra_move_actions": 1,
                            "single_piece_moves": 7,
                            "full_effective_capacity_moves": 8,
                            "under_effective_capacity_moves": 2,
                            "normal_move_effective_capacity_1_moves": 6,
                            "normal_move_effective_capacity_2_moves": 1,
                            "normal_move_effective_capacity_3_moves": 1,
                            "normal_move_effective_capacity_4_moves": 1,
                            "normal_move_effective_capacity_5_moves": 1,
                            "normal_move_capacity_1_moved_1": 6,
                            "normal_move_capacity_2_moved_1": 0,
                            "normal_move_capacity_2_moved_2": 1,
                            "normal_move_capacity_3_moved_1": 1,
                            "normal_move_capacity_3_moved_2": 0,
                            "normal_move_capacity_3_moved_3": 0,
                            "normal_move_capacity_4_moved_1": 0,
                            "normal_move_capacity_4_moved_2": 1,
                            "normal_move_capacity_4_moved_3": 0,
                            "normal_move_capacity_4_moved_4": 0,
                            "normal_move_capacity_5_moved_1": 0,
                            "normal_move_capacity_5_moved_2": 0,
                            "normal_move_capacity_5_moved_3": 0,
                            "normal_move_capacity_5_moved_4": 0,
                            "normal_move_capacity_5_moved_5": 1,
                        },
                    )
                )

            _rows, _series, counts = read_results(path, 100)
            chart = _evaluation_dashboard(counts)
            self.assertIn("Move usage", DASHBOARD_SCRIPT)
            self.assertIn("Avoidable Move1 %", DASHBOARD_SCRIPT)
            self.assertIn("Move capacity utilization by depth", DASHBOARD_SCRIPT)
            self.assertIn("Movement pathology", DASHBOARD_SCRIPT)
            self.assertIn("Move → Claim effectiveness", DASHBOARD_SCRIPT)
            self.assertIn("Movement lesson readiness", DASHBOARD_SCRIPT)
            self.assertNotIn("evaluationLineChart('Move capacity utilization'", DASHBOARD_SCRIPT)
            self.assertNotIn("evaluationLineChart('Move → Claim reward outcomes'", DASHBOARD_SCRIPT)
            self.assertNotIn(
                "evaluationLineChart('Consecutive Move1 inefficiency'", DASHBOARD_SCRIPT
            )
            self.assertIn("Avoidable extra Move actions / 100", DASHBOARD_SCRIPT)
            self.assertNotIn("Full effective-capacity Moves", DASHBOARD_SCRIPT)
            self.assertIn("Avoidable Move1 rate", DASHBOARD_SCRIPT)
            self.assertIn("Pickup #2 vs Placement score gap (training)", DASHBOARD_SCRIPT)
            self.assertIn("Pointless normal Moves / 100", DASHBOARD_SCRIPT)
            self.assertIn("Immediate Q-Undo / 100", DASHBOARD_SCRIPT)
            self.assertNotIn("Repeated-Move penalties/game", DASHBOARD_SCRIPT)
            self.assertNotIn("All-Move-turn penalties/game", DASHBOARD_SCRIPT)
            self.assertIn("Pickup #2 is Q1", DASHBOARD_SCRIPT)
            self.assertIn("Pickup #3+ continuation is Q1", DASHBOARD_SCRIPT)
            movement_titles = (
                "Move usage",
                "Avoidable Move1 %",
                "Move capacity utilization by depth",
                "Movement pathology",
                "Move → Claim effectiveness",
                "Movement lesson readiness",
            )
            self.assertEqual(
                [DASHBOARD_SCRIPT.index(title) for title in movement_titles],
                sorted(DASHBOARD_SCRIPT.index(title) for title in movement_titles),
            )
            self.assertNotIn("Pointless Move workflows per game", chart)
            self.assertEqual(chart.count("data-evaluation-panel"), 1)
            self.assertIn('"move_action_count":30.0', chart)
            self.assertIn('"pointless_move_workflows":6.0', chart)
            self.assertIn('"pointless_normal_move_workflows":3.0', chart)
            self.assertIn('"immediate_one_piece_q_undos":3.0', chart)
            self.assertIn('"single_piece_moves":19.0', chart)
            self.assertIn('"single_piece_moves_creating_claimable_route":4.0', chart)
            self.assertIn('"single_piece_move_claim_conversions":2.0', chart)
            self.assertIn('"full_effective_capacity_moves":21.0', chart)
            self.assertIn('"under_effective_capacity_moves":9.0', chart)
            self.assertIn('"normal_move_effective_capacity_3_moves":4.0', chart)
            self.assertIn('"normal_move_capacity_3_moved_3":1.0', chart)
            self.assertIn('"normal_move_effective_capacity_5_moves":3.0', chart)
            self.assertIn('"normal_move_capacity_5_moved_5":1.0', chart)
            self.assertIn('"move_claim_reward_awarded":3.0', chart)
            self.assertIn('"move_claim_reward_blocked_already_claimable":3.0', chart)
            self.assertIn('"consecutive_move1_pairs":7.0', chart)
            self.assertIn('"consecutive_move1_pairs_with_multiple_available":5.0', chart)
            self.assertIn('"consecutive_move1_pairs_move4_capacity":3.0', chart)
            self.assertIn('"avoidable_extra_move_actions":4.0', chart)
            self.assertIn('"pointless_move_any2_workflows":3.0', chart)
            self.assertIn('"map":"2","players":"3"', chart)

    def test_evaluation_types_share_one_filterable_dashboard_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            fieldnames = (
                "game#",
                "batch#",
                "run",
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
                            "run": "evaluation_mid_late_end",
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
                            "run": "evaluation_fresh",
                            "evaluation_suite_version": 10,
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
            self.assertIn('<option value="fresh">Fresh</option>', dashboard)
            self.assertNotIn("Mixed Development", dashboard)
            self.assertIn("data-evaluation-map", dashboard)
            self.assertIn("data-evaluation-players", dashboard)
            self.assertIn("datasets[mode]", DASHBOARD_SCRIPT)
            self.assertIn("data-evaluation-title", DASHBOARD_SCRIPT)
            self.assertIn("Tier 1 win rate by player count", DASHBOARD_SCRIPT)
            self.assertIn("Average interactions per game", DASHBOARD_SCRIPT)
            self.assertIn("Average paid actions per game", DASHBOARD_SCRIPT)
            self.assertIn("Interactions/game", DASHBOARD_SCRIPT)
            self.assertIn("Paid actions/game", DASHBOARD_SCRIPT)
            self.assertIn("Fresh-game interaction-limit rate", DASHBOARD_SCRIPT)
            self.assertNotIn("Average game length", DASHBOARD_SCRIPT)
            self.assertIn("Move usage", DASHBOARD_SCRIPT)
            self.assertIn("Move → Claim effectiveness", DASHBOARD_SCRIPT)
            self.assertIn('"standard":', dashboard)
            self.assertIn('"fresh":', dashboard)
            self.assertIn('"map":"2","players":"5"', dashboard)

            self.assertIn("Movement lesson readiness", DASHBOARD_SCRIPT)

    def test_movement_readiness_uses_weighted_training_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            fieldnames = (
                "game#",
                "batch#",
                "run",
                "map",
                "player_count",
                "completion_reason",
                "case_a_family_ranking_samples",
                "case_a_family_ranking_q1_pickup_fraction",
                "move_continuation_family_ranking_samples",
                "move_continuation_q1_pickup_fraction",
                "move1_scaffold_mask_states",
                "move1_scaffold_best_pickup_minus_best_placement_mean",
            )
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "game#": 1,
                            "batch#": 4,
                            "run": "training_fresh",
                            "map": 2,
                            "player_count": 3,
                            "completion_reason": "20_points",
                            "case_a_family_ranking_samples": 10,
                            "case_a_family_ranking_q1_pickup_fraction": 0.5,
                            "move_continuation_family_ranking_samples": 4,
                            "move_continuation_q1_pickup_fraction": 0.25,
                            "move1_scaffold_mask_states": 2,
                            "move1_scaffold_best_pickup_minus_best_placement_mean": 1,
                        },
                        {
                            "game#": 2,
                            "batch#": 4,
                            "run": "training_mid",
                            "map": 2,
                            "player_count": 3,
                            "completion_reason": "20_points",
                            "case_a_family_ranking_samples": 30,
                            "case_a_family_ranking_q1_pickup_fraction": 0.75,
                            "move_continuation_family_ranking_samples": 6,
                            "move_continuation_q1_pickup_fraction": 0.5,
                            "move1_scaffold_mask_states": 3,
                            "move1_scaffold_best_pickup_minus_best_placement_mean": 3,
                        },
                        {
                            "game#": 3,
                            "batch#": 4,
                            "run": "evaluation_fresh",
                            "map": 2,
                            "player_count": 3,
                            "completion_reason": "20_points",
                        },
                    )
                )

            _rows, _series, counts = read_results(path, 100)
            dashboard = _evaluation_dashboard(counts)

            self.assertIn('"trainingMovementRecords":[', dashboard)
            self.assertIn('"case_a_family_ranking_q1_pickup_fraction":27.5', dashboard)
            self.assertIn('"case_a_family_ranking_q1_pickup_fraction":40.0', dashboard)
            self.assertIn('"move_continuation_q1_pickup_fraction":4.0', dashboard)
            self.assertIn('"move_continuation_q1_pickup_fraction":10.0', dashboard)
            self.assertIn(
                '"move1_scaffold_best_pickup_minus_best_placement_mean":11.0',
                dashboard,
            )
            self.assertIn(
                '"move1_scaffold_best_pickup_minus_best_placement_mean":5.0',
                dashboard,
            )

    def test_movement_charts_preserve_no_data_for_missing_or_zero_denominators(self):
        self.assertIn("denominator > 0", DASHBOARD_SCRIPT)
        self.assertIn("const movementRatioToSum", DASHBOARD_SCRIPT)
        self.assertIn("const avoidableMove1Ratio", DASHBOARD_SCRIPT)
        self.assertIn(
            "entry.movementTotals[totalNumerator] - entry.movementTotals[partNumerator]",
            DASHBOARD_SCRIPT,
        )
        self.assertIn(
            "entry.movementTotals[totalDenominator] - entry.movementTotals[partDenominator]",
            DASHBOARD_SCRIPT,
        )
        self.assertIn("No data available.", DASHBOARD_SCRIPT)
        self.assertIn("if (!Number.isFinite(value)) { continuing = false", DASHBOARD_SCRIPT)

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
            self.assertIn("Interaction-limit terminations</strong><span>1", dashboard)
            self.assertNotIn("Completion results", dashboard)
            self.assertNotIn("Game types", dashboard)


if __name__ == "__main__":
    unittest.main()
