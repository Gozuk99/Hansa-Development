from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from ai.ai_model import HansaNN, device
from ai.observation_schema import OBSERVATION_SIZE
from game.action_schema import ACTION_SPACE_SIZE
from training.self_play import (
    ActionSelection,
    CompletedTrajectory,
    ShadowFilterAudit,
    ShadowPolicyMetrics,
    SelfPlayTrainer,
    TrainingConfig,
    TrainingDecision,
    apply_local_training_targets,
    assign_reward_to_go,
    policy_decision_loss,
    policy_quality_signal,
    record_shadow_policy_metrics,
    semantic_group_logits,
    would_shadow_filter,
)
from tests.action_helpers import self_play_test_state


STATE = self_play_test_state()


def decision(
    *,
    target=500.0,
    selected=0,
    equivalent_action_indices=(),
    equivalent_action_groups=(),
):
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.uint8)
    mask[:3] = 1
    return TrainingDecision(
        observation=torch.zeros(OBSERVATION_SIZE),
        legal_action_mask=mask,
        action_index=selected,
        acting_player_index=0,
        player_reward_deltas=(target, 0.0, 0.0),
        immediate_reward=target,
        policy_tier=1,
        epsilon=0.05,
        top_k=2,
        used_epsilon=False,
        model_rank=1,
        legal_action_count=3,
        reward_to_go=target,
        equivalent_action_indices=equivalent_action_indices,
        equivalent_action_groups=equivalent_action_groups,
    )


def trajectory(sample):
    return CompletedTrajectory(
        decisions=(sample,),
        terminal_rewards=(0.0, 0.0, 0.0),
        final_scores=(0, 0, 0),
        winner_indices=(),
        action_trace=(sample.action_index,),
        seat_tiers=(1, 3, 5),
    )


class PolicyHeadTrainingTests(unittest.TestCase):
    def test_quality_signal_is_signed_and_bounded(self):
        targets = torch.tensor((-100_000.0, 0.0, 100_000.0))
        quality = policy_quality_signal(targets, 1_000.0)

        self.assertLess(quality[0], 0)
        self.assertEqual(quality[1], 0)
        self.assertGreater(quality[2], 0)
        self.assertTrue(torch.all(quality.abs() <= 1))

    def test_positive_and_negative_quality_move_preference_in_opposite_directions(self):
        sample = decision()
        positive_logits = torch.zeros(ACTION_SPACE_SIZE, requires_grad=True)
        positive_loss = policy_decision_loss(positive_logits, sample, torch.tensor(1.0))
        positive_loss.backward()
        negative_logits = torch.zeros(ACTION_SPACE_SIZE, requires_grad=True)
        negative_loss = policy_decision_loss(negative_logits, sample, torch.tensor(-1.0))
        negative_loss.backward()

        self.assertLess(positive_logits.grad[0], 0)
        self.assertGreater(negative_logits.grad[0], 0)
        baseline_probability = 1 / 3
        positive_updated = torch.softmax((positive_logits - positive_logits.grad)[:3], dim=0)[0]
        negative_updated = torch.softmax((negative_logits - negative_logits.grad)[:3], dim=0)[0]
        self.assertGreater(positive_updated, baseline_probability)
        self.assertLess(negative_updated, baseline_probability)

    def test_zero_quality_has_zero_policy_gradient(self):
        logits = torch.randn(ACTION_SPACE_SIZE, requires_grad=True)

        loss = policy_decision_loss(logits, decision(), torch.tensor(0.0))
        loss.backward()

        self.assertEqual(loss, 0)
        self.assertEqual(torch.count_nonzero(logits.grad), 0)

    def test_negative_quality_loss_is_finite_and_gradient_fades_at_low_probability(self):
        gradients = []
        for selected_logit in (-5.0, -20.0):
            logits = torch.zeros(ACTION_SPACE_SIZE, requires_grad=True)
            with torch.no_grad():
                logits[0] = selected_logit
            loss = policy_decision_loss(logits, decision(), torch.tensor(-1.0))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreaterEqual(loss, 0)
            gradients.append(abs(float(logits.grad[0])))

        self.assertLess(gradients[1], gradients[0])
        self.assertLess(gradients[1], 1e-6)

    def test_single_legal_semantic_choice_has_zero_policy_loss(self):
        sample = decision()
        sample.legal_action_mask.zero_()
        sample.legal_action_mask[sample.action_index] = 1
        for quality in (-1.0, 1.0):
            logits = torch.randn(ACTION_SPACE_SIZE, requires_grad=True)
            loss = policy_decision_loss(logits, sample, torch.tensor(quality))
            loss.backward()
            self.assertEqual(loss, 0)
            self.assertEqual(torch.count_nonzero(logits.grad), 0)

    def test_illegal_logits_are_excluded_from_policy_normalization(self):
        sample = decision()
        baseline = torch.zeros(ACTION_SPACE_SIZE)
        illegal_high = baseline.clone()
        illegal_high[100] = 1_000

        baseline_loss = policy_decision_loss(baseline, sample, torch.tensor(1.0))
        illegal_loss = policy_decision_loss(illegal_high, sample, torch.tensor(1.0))

        self.assertEqual(baseline_loss, illegal_loss)

    def test_equivalent_selected_indices_form_one_semantic_choice(self):
        sample = decision(
            selected=1,
            equivalent_action_indices=(0, 1),
            equivalent_action_groups=((0, 1),),
        )
        logits = torch.zeros(ACTION_SPACE_SIZE, requires_grad=True)
        loss = policy_decision_loss(logits, sample, torch.tensor(1.0))
        loss.backward()

        self.assertAlmostEqual(loss.item(), -torch.log(torch.tensor(1 / 2)).item(), places=6)
        self.assertEqual(logits.grad[0], logits.grad[1])
        self.assertLess(logits.grad[0], 0)

    def test_equivalent_index_multiplicity_does_not_change_policy_probability(self):
        grouped = decision(
            selected=1,
            equivalent_action_indices=(0, 1),
            equivalent_action_groups=((0, 1),),
        )
        singleton = decision(selected=0)
        singleton_mask = singleton.legal_action_mask.clone()
        singleton_mask[2] = 0
        singleton = TrainingDecision(
            **{
                **asdict(singleton),
                "legal_action_mask": singleton_mask,
                "observation": singleton.observation,
            }
        )
        logits = torch.zeros(ACTION_SPACE_SIZE)

        grouped_loss = policy_decision_loss(logits, grouped, torch.tensor(1.0))
        singleton_loss = policy_decision_loss(logits, singleton, torch.tensor(1.0))

        self.assertAlmostEqual(grouped_loss.item(), singleton_loss.item(), places=6)

    def test_parallel_update_changes_q_trunk_and_policy_parameters(self):
        trainer = SelfPlayTrainer(config=TrainingConfig(decision_batch_size=1))
        before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}

        trainer.update_model((trajectory(decision()),))

        changed = {
            name
            for name, value in trainer.model.named_parameters()
            if not torch.equal(before[name], value)
        }
        self.assertIn("layer1.bias", changed)
        self.assertIn("layer3.weight", changed)
        self.assertIn("policy_head.weight", changed)
        self.assertIsNotNone(trainer.progress.last_q_loss)
        self.assertIsNotNone(trainer.progress.last_policy_loss)
        self.assertIsNotNone(trainer.progress.last_total_loss)

    def test_policy_backward_changes_only_policy_head(self):
        model = HansaNN()
        observation = torch.ones((1, OBSERVATION_SIZE))
        output = model(observation)
        policy_loss = -torch.log_softmax(output.policy_logits[0, :3], dim=0)[0]

        model.zero_grad()
        policy_loss.backward()

        for parameter in (
            *model.layer1.parameters(),
            *model.layer2.parameters(),
            *model.shared_layer3.parameters(),
        ):
            self.assertIsNone(parameter.grad)
        for parameter in model.layer3.parameters():
            self.assertIsNone(parameter.grad)
        self.assertGreater(torch.count_nonzero(model.policy_head.weight.grad), 0)

    def test_policy_only_optimizer_step_changes_only_policy_head(self):
        trainer = SelfPlayTrainer()
        observation = torch.ones((1, OBSERVATION_SIZE))
        output = trainer.model(observation)
        q_loss = output.q_values.sum() * 0
        policy_loss = -torch.log_softmax(output.policy_logits[0, :3], dim=0)[0]
        before = {
            name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
        }

        trainer._backward_independent_losses(q_loss, policy_loss)
        trainer.optimizer.step()

        for name, parameter in trainer.model.named_parameters():
            changed = not torch.equal(before[name], parameter)
            self.assertEqual(changed, name.startswith("policy_head."), name)

    def test_combined_shadow_update_matches_q_only_q_and_trunk_update(self):
        q_only = SelfPlayTrainer(
            config=TrainingConfig(decision_batch_size=1, policy_loss_weight=0.0)
        )
        combined = SelfPlayTrainer(config=TrainingConfig(decision_batch_size=1))
        combined.model.load_state_dict(q_only.model.state_dict())
        training_trajectory = trajectory(decision())

        q_only.update_model((training_trajectory,))
        combined.update_model((training_trajectory,))

        for name, q_only_parameter in q_only.model.named_parameters():
            if name.startswith("policy_head."):
                continue
            combined_parameter = dict(combined.model.named_parameters())[name]
            self.assertTrue(torch.equal(q_only_parameter, combined_parameter), name)
        q_only_optimizer = q_only.optimizer.state_dict()
        combined_optimizer = combined.optimizer.state_dict()
        for q_id, combined_id in zip(
            q_only_optimizer["param_groups"][0]["params"],
            combined_optimizer["param_groups"][0]["params"],
        ):
            for state_name, expected in q_only_optimizer["state"][q_id].items():
                actual = combined_optimizer["state"][combined_id][state_name]
                if torch.is_tensor(expected):
                    self.assertTrue(torch.equal(expected, actual), state_name)
                else:
                    self.assertEqual(expected, actual)

    def test_large_policy_gradient_cannot_change_q_or_trunk_gradients(self):
        config = TrainingConfig(max_gradient_norm=0.01, policy_loss_weight=1_000_000.0)
        baseline = SelfPlayTrainer(config=config)
        joint = SelfPlayTrainer(config=config)
        joint.model.load_state_dict(baseline.model.state_dict())
        observations = torch.ones((1, OBSERVATION_SIZE))

        baseline_output = baseline.model(observations)
        baseline_q_loss = baseline_output.q_values.square().mean()
        baseline.optimizer.zero_grad(set_to_none=True)
        baseline_q_loss.backward()
        baseline._clip_q_gradients()
        expected = {
            name: parameter.grad.detach().clone()
            for name, parameter in baseline.model.named_parameters()
            if not name.startswith("policy_head.") and parameter.grad is not None
        }

        joint_output = joint.model(observations)
        joint_q_loss = joint_output.q_values.square().mean()
        joint_policy_loss = -torch.log_softmax(joint_output.policy_logits[0, :3], dim=0)[0]
        joint._backward_independent_losses(joint_q_loss, joint_policy_loss)

        for name, expected_gradient in expected.items():
            self.assertTrue(
                torch.equal(dict(joint.model.named_parameters())[name].grad, expected_gradient),
                name,
            )
        self.assertGreater(torch.count_nonzero(joint.model.policy_head.weight.grad), 0)

    def test_policy_head_learning_rate_matches_q_while_trunk_scale_stays_zero(self):
        trainer = SelfPlayTrainer()

        self.assertEqual(trainer._policy_trunk_gradient_scale(), 0.0)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.0001)
        self.assertAlmostEqual(trainer.optimizer.param_groups[1]["lr"], 0.0001)

    def test_dual_head_checkpoint_preserves_policy_age_and_zero_trunk_influence(self):
        trainer = SelfPlayTrainer()
        trainer.update_model((trajectory(decision()),))
        trainer.progress.training_updates = 12_345
        trainer.progress.policy_training_updates = 5_000
        with torch.no_grad():
            trainer.model.policy_head.weight.fill_(0.125)
            trainer.model.policy_head.bias.fill_(-0.25)
        expected_policy = {
            name: value.detach().clone()
            for name, value in trainer.model.state_dict().items()
            if name.startswith("policy_head.")
        }
        expected_optimizer = trainer.optimizer.state_dict()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dual-head.pth"
            trainer.save_checkpoint(path, ())
            restored = SelfPlayTrainer.from_checkpoint(path)

        self.assertEqual(restored.progress.training_updates, 12_345)
        self.assertEqual(restored.progress.policy_training_updates, 5_000)
        self.assertEqual(restored._policy_trunk_gradient_scale(), 0.0)
        for name, expected in expected_policy.items():
            self.assertTrue(torch.equal(restored.model.state_dict()[name], expected), name)
        restored_optimizer = restored.optimizer.state_dict()
        for expected_group, restored_group in zip(
            expected_optimizer["param_groups"], restored_optimizer["param_groups"]
        ):
            self.assertEqual(expected_group, restored_group)
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for state_name, expected in expected_state.items():
                actual = restored_optimizer["state"][parameter_id][state_name]
                if torch.is_tensor(expected):
                    self.assertTrue(
                        torch.equal(expected.detach().cpu(), actual.detach().cpu()),
                        state_name,
                    )
                else:
                    self.assertEqual(expected, actual)

    def test_policy_head_does_not_change_seeded_q_gameplay_trace(self):
        first = SelfPlayTrainer(config=TrainingConfig(seed=777, max_actions=100))
        second = SelfPlayTrainer(config=TrainingConfig(seed=777, max_actions=100))
        with torch.no_grad():
            second.model.policy_head.weight.fill_(99)
            second.model.policy_head.bias.fill_(-99)

        first_trajectory = first.collect_game(STATE)
        second_trajectory = second.collect_game(STATE)

        self.assertEqual(first_trajectory.action_trace, second_trajectory.action_trace)
        self.assertEqual(first_trajectory.final_scores, second_trajectory.final_scores)

    def test_identity_shared_layer_preserves_seeded_top_one_gameplay_trace(self):
        config = TrainingConfig(
            seed=778,
            max_actions=100,
            tier_top_k=(1, 1, 1, 1, 1),
            tier_epsilons=(0, 0, 0, 0, 0),
        )
        legacy = SelfPlayTrainer(config=config)
        migrated = SelfPlayTrainer(config=config)
        migrated.model.load_state_dict(legacy.model.state_dict())

        with mock.patch.object(
            legacy.model.shared_layer3,
            "forward",
            side_effect=lambda features: features,
        ):
            legacy_trajectory = legacy.collect_game(STATE)
        migrated_trajectory = migrated.collect_game(STATE)

        self.assertEqual(legacy_trajectory.action_trace, migrated_trajectory.action_trace)
        self.assertEqual(legacy_trajectory.final_scores, migrated_trajectory.final_scores)

    def test_shadow_metrics_match_reference_semantic_calculation(self):
        groups = ((0, 1), (2,), (3,), (4,), (5,), (6,))
        q_group_scores = (5.0, 4.0, 4.0, 2.0, 1.0, 0.0)
        logits = torch.zeros(ACTION_SPACE_SIZE, device=device)
        logits[:7] = torch.tensor((2.0, 0.0, 2.0, 0.0, -1.0, 3.0, 0.5), device=device)
        metrics = ShadowPolicyMetrics()

        record_shadow_policy_metrics(metrics, q_group_scores, logits, groups)
        actual = metrics.averages()

        group_logits = semantic_group_logits(logits, groups)
        probabilities = torch.softmax(group_logits, dim=0)
        q_ranking = sorted(
            range(len(q_group_scores)),
            key=lambda index: (-q_group_scores[index], index),
        )
        policy_top = int(torch.argmax(probabilities).cpu())
        sorted_probabilities = torch.sort(probabilities, descending=True).values
        expected = (
            float(policy_top == q_ranking[0]),
            float(q_ranking.index(policy_top) + 1),
            float((-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()).cpu()),
            float(sorted_probabilities[:2].sum().cpu()),
            float(sorted_probabilities[:5].sum().cpu()),
            float(sorted_probabilities[:10].sum().cpu()),
        )

        for observed, reference in zip(actual, expected):
            self.assertAlmostEqual(observed, reference, places=6)

    def test_shadow_diagnostics_do_not_change_seeded_gameplay_trace(self):
        enabled = SelfPlayTrainer(config=TrainingConfig(seed=881, max_actions=100))
        disabled = SelfPlayTrainer(config=TrainingConfig(seed=881, max_actions=100))

        enabled_trajectory = enabled.collect_game(STATE)
        with mock.patch("training.self_play.record_shadow_policy_metrics"):
            disabled_trajectory = disabled.collect_game(STATE)

        self.assertEqual(enabled_trajectory.action_trace, disabled_trajectory.action_trace)
        self.assertEqual(enabled_trajectory.final_scores, disabled_trajectory.final_scores)

    def test_shadow_filter_requires_both_semantic_rank_gates(self):
        self.assertTrue(would_shadow_filter(11, 21))
        self.assertFalse(would_shadow_filter(11, 20))
        self.assertFalse(would_shadow_filter(10, 21))
        self.assertFalse(would_shadow_filter(1, 1))

    def test_shadow_filter_groups_equivalent_actions_and_attaches_final_targets(self):
        groups = tuple((index,) for index in range(3, 23)) + ((0, 1, 2),)
        q_scores = tuple(float(value) for value in range(21, 0, -1))
        selection = ActionSelection(
            action_index=1,
            used_epsilon=True,
            model_rank=21,
            legal_action_count=21,
            equivalent_action_indices=(0, 1, 2),
            semantic_q_scores=q_scores,
        )
        audit = ShadowFilterAudit()
        audit.record(
            0,
            selection,
            torch.zeros(ACTION_SPACE_SIZE, device=device),
            groups,
        )
        sample = decision(selected=1, equivalent_action_indices=(0, 1, 2))
        sample = replace(
            sample,
            player_reward_deltas=(5.0, 0.0, 0.0),
            immediate_reward=5.0,
            local_training_adjustment=-2.0,
            reward_to_go=None,
            used_epsilon=True,
        )
        reward_decisions = assign_reward_to_go((sample,), (100.0, 0.0, 0.0), 0.99)
        training_decisions = apply_local_training_targets(reward_decisions)

        records = audit.flagged_outcomes(
            reward_decisions,
            training_decisions,
            (100.0, 0.0, 0.0),
            (30, 20, 10),
            (0,),
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.semantic_action_indices, (0, 1, 2))
        self.assertEqual(record.semantic_q_rank, 21)
        self.assertEqual(record.semantic_policy_rank, 21)
        self.assertAlmostEqual(record.policy_probability, 1 / 21, places=6)
        self.assertEqual(record.reward_to_go, 105.0)
        self.assertEqual(record.final_training_target, 103.0)
        self.assertEqual(record.local_training_adjustment, -2.0)
        self.assertTrue(record.receives_terminal_credit)
        self.assertEqual(record.terminal_credit_value, 100.0)
        self.assertEqual(record.acting_player_final_score, 30)
        self.assertTrue(record.acting_player_won)
        self.assertTrue(record.used_epsilon)

    def test_shadow_filter_records_only_selected_actions_that_cross_both_gates(self):
        groups = tuple((index,) for index in range(21))
        logits = torch.arange(21, 0, -1, dtype=torch.float32, device=device)
        audit = ShadowFilterAudit()
        audit.record(
            0,
            ActionSelection(20, False, 21, 21, (20,), tuple(logits.cpu().tolist())),
            logits,
            groups,
        )
        audit.record(
            1,
            ActionSelection(0, False, 1, 21, (0,), tuple(logits.cpu().tolist())),
            logits,
            groups,
        )
        samples = (decision(selected=20), decision(selected=0))
        reward_decisions = assign_reward_to_go(samples, (0.0, 0.0, 0.0), 0.99)
        records = audit.flagged_outcomes(
            reward_decisions,
            apply_local_training_targets(reward_decisions),
            (0.0, 0.0, 0.0),
            (0, 0, 0),
            (),
        )

        self.assertEqual([record.action_index for record in records], [20])

    def test_shadow_filter_audit_does_not_change_trace_and_skips_evaluation(self):
        enabled = SelfPlayTrainer(
            config=TrainingConfig(
                seed=991,
                max_actions=100,
                shadow_filter_audit_enabled=True,
            )
        )
        disabled = SelfPlayTrainer(config=TrainingConfig(seed=991, max_actions=100))

        enabled_trajectory = enabled.collect_game(STATE)
        disabled_trajectory = disabled.collect_game(STATE)
        evaluation = SelfPlayTrainer(
            config=TrainingConfig(
                seed=991,
                max_actions=100,
                shadow_filter_audit_enabled=True,
            )
        ).collect_game(
            STATE,
            evaluation=True,
            capture_action_limit=True,
        )

        self.assertEqual(enabled_trajectory.action_trace, disabled_trajectory.action_trace)
        self.assertEqual(enabled_trajectory.final_scores, disabled_trajectory.final_scores)
        self.assertEqual(evaluation.shadow_filter_records, ())
        self.assertEqual(evaluation.shadow_filter_selected_count, 0)

    def test_shadow_filter_audit_is_disabled_by_default_without_hot_path_calculation(self):
        trainer = SelfPlayTrainer(config=TrainingConfig(seed=991, max_actions=100))

        with (
            mock.patch.object(ShadowFilterAudit, "record") as record,
            mock.patch("training.self_play.semantic_group_logits") as group_logits,
        ):
            trajectory = trainer.collect_game(STATE)

        self.assertFalse(trainer.config.shadow_filter_audit_enabled)
        record.assert_not_called()
        group_logits.assert_not_called()
        self.assertEqual(trajectory.shadow_filter_records, ())
        self.assertEqual(trajectory.shadow_filter_selected_count, 0)
        self.assertEqual(trajectory.shadow_filter_epsilon_selected_count, 0)

    def test_shadow_filter_flag_does_not_change_q_or_policy_training(self):
        enabled = SelfPlayTrainer(config=TrainingConfig(seed=992, max_actions=100))
        disabled = SelfPlayTrainer(config=TrainingConfig(seed=992, max_actions=100))

        enabled_trajectory = enabled.collect_game(STATE, shadow_filter_audit=True)
        disabled_trajectory = disabled.collect_game(STATE, shadow_filter_audit=False)
        enabled_loss = enabled.update_model((enabled_trajectory,))
        disabled_loss = disabled.update_model((disabled_trajectory,))

        self.assertEqual(enabled_trajectory.action_trace, disabled_trajectory.action_trace)
        self.assertEqual(enabled_loss, disabled_loss)
        self.assertEqual(enabled.progress.last_q_loss, disabled.progress.last_q_loss)
        self.assertEqual(enabled.progress.last_policy_loss, disabled.progress.last_policy_loss)
        for enabled_value, disabled_value in zip(
            enabled.model.state_dict().values(),
            disabled.model.state_dict().values(),
        ):
            self.assertTrue(torch.equal(enabled_value, disabled_value))

    def test_checkpoint_without_shadow_filter_flag_resumes_without_resetting_state(self):
        trainer = SelfPlayTrainer(config=TrainingConfig(seed=993))
        trainer.update_model((trajectory(decision()),))
        expected_model = {
            name: value.detach().clone() for name, value in trainer.model.state_dict().items()
        }
        expected_optimizer = trainer.optimizer.state_dict()
        expected_progress = trainer.progress

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre-shadow-flag.pth"
            trainer.save_checkpoint(path, ())
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["training_config"].pop("shadow_filter_audit_enabled")
            torch.save(checkpoint, path)
            restored = SelfPlayTrainer.from_checkpoint(path)

        self.assertFalse(restored.config.shadow_filter_audit_enabled)
        self.assertEqual(restored.progress.training_updates, expected_progress.training_updates)
        self.assertEqual(
            restored.progress.policy_training_updates,
            expected_progress.policy_training_updates,
        )
        for name, expected in expected_model.items():
            self.assertTrue(torch.equal(restored.model.state_dict()[name], expected), name)
        restored_optimizer = restored.optimizer.state_dict()
        self.assertEqual(restored_optimizer["param_groups"], expected_optimizer["param_groups"])
        for parameter_id, expected_state in expected_optimizer["state"].items():
            for state_name, expected in expected_state.items():
                actual = restored_optimizer["state"][parameter_id][state_name]
                if torch.is_tensor(expected):
                    self.assertTrue(
                        torch.equal(expected.detach().cpu(), actual.detach().cpu()),
                        state_name,
                    )
                else:
                    self.assertEqual(expected, actual)

    def test_q_only_training_checkpoint_migrates_model_and_optimizer(self):
        trainer = SelfPlayTrainer()
        trainer.update_model((trajectory(decision()),))
        q_state = {
            key: value.clone()
            for key, value in trainer.model.state_dict().items()
            if not key.startswith(("policy_head.", "shared_layer3."))
        }
        optimizer_state = trainer.optimizer.state_dict()
        current_q_parameter_ids = optimizer_state["param_groups"][0]["params"]
        shared_parameter_ids = {
            current_q_parameter_ids[index]
            for index, parameter in enumerate(trainer._q_and_trunk_parameters)
            if any(parameter is shared for shared in trainer.model.shared_layer3.parameters())
        }
        q_parameter_ids = [
            parameter_id
            for parameter_id in current_q_parameter_ids
            if parameter_id not in shared_parameter_ids
        ]
        legacy_optimizer = {
            "state": {
                parameter_id: optimizer_state["state"][parameter_id]
                for parameter_id in q_parameter_ids
            },
            "param_groups": [{**optimizer_state["param_groups"][0], "params": q_parameter_ids}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            trainer.save_checkpoint(path, ())
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["training_checkpoint_version"] = 5
            checkpoint.pop("model_checkpoint_format", None)
            checkpoint.pop("model_checkpoint_version", None)
            checkpoint["state_dict"] = q_state
            checkpoint["optimizer_state_dict"] = legacy_optimizer
            checkpoint["training_progress"]["training_updates"] = 12_345
            checkpoint["training_progress"].pop("policy_training_updates", None)
            checkpoint["training_config"] = {
                key: value
                for key, value in asdict(trainer.config).items()
                if not key.startswith("policy_")
            }
            torch.save(checkpoint, path)

            restored = SelfPlayTrainer.from_checkpoint(path)

        self.assertTrue(restored.model.migrated_policy_head)
        self.assertEqual(restored.progress.training_updates, 12_345)
        self.assertEqual(restored.progress.policy_training_updates, 0)
        self.assertEqual(restored._policy_trunk_gradient_scale(), 0.0)
        for key, expected in q_state.items():
            self.assertTrue(torch.equal(restored.model.state_dict()[key], expected), key)
        self.assertEqual(torch.count_nonzero(restored.model.policy_head.weight), 0)
        self.assertEqual(torch.count_nonzero(restored.model.policy_head.bias), 0)
        with torch.no_grad():
            policy_logits = restored.model(torch.ones((1, OBSERVATION_SIZE))).policy_logits
        self.assertTrue(torch.equal(policy_logits, torch.zeros_like(policy_logits)))
        self.assertEqual(len(restored.optimizer.param_groups), 2)
        self.assertEqual(
            len(restored.optimizer.state_dict()["state"]),
            len(q_parameter_ids) + 2,
        )
        restored_optimizer = restored.optimizer.state_dict()
        restored_q_parameter_ids = restored_optimizer["param_groups"][0]["params"]
        for old_id, restored_id in zip(q_parameter_ids, restored_q_parameter_ids):
            for state_name, expected in legacy_optimizer["state"][old_id].items():
                actual = restored_optimizer["state"][restored_id][state_name]
                if torch.is_tensor(expected):
                    self.assertTrue(
                        torch.equal(actual.detach().cpu(), expected.detach().cpu()),
                        state_name,
                    )
                else:
                    self.assertEqual(actual, expected)
        for restored_id in restored_q_parameter_ids[len(q_parameter_ids) :]:
            state = restored_optimizer["state"][restored_id]
            self.assertEqual(state["step"].item(), 0)
            self.assertFalse(state["exp_avg"].any())
            self.assertFalse(state["exp_avg_sq"].any())

        restored.update_model((trajectory(decision()),))
        self.assertEqual(restored.progress.training_updates, 12_346)
        self.assertEqual(restored.progress.policy_training_updates, 1)

    def test_dual_head_checkpoint_adds_neutral_shared_layer_optimizer_state(self):
        trainer = SelfPlayTrainer()
        trainer.update_model((trajectory(decision()),))
        optimizer_state = trainer.optimizer.state_dict()
        q_ids = tuple(optimizer_state["param_groups"][0]["params"])
        shared_indices = tuple(
            index
            for index, parameter in enumerate(trainer._q_and_trunk_parameters)
            if any(parameter is shared for shared in trainer.model.shared_layer3.parameters())
        )
        legacy_q_ids = tuple(
            parameter_id for index, parameter_id in enumerate(q_ids) if index not in shared_indices
        )
        legacy_optimizer = {
            "state": {
                parameter_id: state
                for parameter_id, state in optimizer_state["state"].items()
                if parameter_id not in {q_ids[index] for index in shared_indices}
            },
            "param_groups": [
                {**optimizer_state["param_groups"][0], "params": list(legacy_q_ids)},
                optimizer_state["param_groups"][1],
            ],
        }
        legacy_model = {
            key: value.clone()
            for key, value in trainer.model.state_dict().items()
            if not key.startswith("shared_layer3.")
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-dual-head.pth"
            trainer.save_checkpoint(path, ())
            checkpoint = torch.load(path, map_location="cpu")
            checkpoint["training_checkpoint_version"] = 6
            checkpoint["model_checkpoint_version"] = 2
            checkpoint["state_dict"] = legacy_model
            checkpoint["optimizer_state_dict"] = legacy_optimizer
            torch.save(checkpoint, path)
            restored = SelfPlayTrainer.from_checkpoint(path)

        self.assertTrue(restored.model.migrated_shared_layer)
        self.assertTrue(torch.equal(restored.model.shared_layer3.weight.cpu(), torch.eye(1024)))
        self.assertFalse(restored.model.shared_layer3.bias.any())
        for key, expected in legacy_model.items():
            self.assertTrue(
                torch.equal(restored.model.state_dict()[key].cpu(), expected.cpu()),
                key,
            )

        restored_optimizer = restored.optimizer.state_dict()
        restored_q_ids = tuple(restored_optimizer["param_groups"][0]["params"])
        for old_id, new_id in zip(legacy_q_ids, restored_q_ids):
            for state_name, expected in legacy_optimizer["state"][old_id].items():
                actual = restored_optimizer["state"][new_id][state_name]
                self.assertTrue(torch.equal(actual.cpu(), expected.cpu()), state_name)
        old_policy_ids = tuple(legacy_optimizer["param_groups"][1]["params"])
        new_policy_ids = tuple(restored_optimizer["param_groups"][1]["params"])
        for old_id, new_id in zip(old_policy_ids, new_policy_ids):
            for state_name, expected in legacy_optimizer["state"][old_id].items():
                actual = restored_optimizer["state"][new_id][state_name]
                self.assertTrue(torch.equal(actual.cpu(), expected.cpu()), state_name)
        for index in shared_indices:
            state = restored_optimizer["state"][restored_q_ids[index]]
            self.assertEqual(state["step"].item(), 0)
            self.assertFalse(state["exp_avg"].any())
            self.assertFalse(state["exp_avg_sq"].any())
        self.assertAlmostEqual(
            restored.progress.mean_policy_loss,
            restored.progress.last_policy_loss,
        )


if __name__ == "__main__":
    unittest.main()
