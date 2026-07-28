from dataclasses import dataclass
import random

from ai.action_options import masking_out_invalid_actions
from game.game_info import Game
from game.invariants import validate_game


@dataclass(frozen=True)
class GameRunResult:
    map_num: int
    num_players: int
    seed: int
    action_count: int
    terminal_reason: str
    final_scores: tuple
    action_trace: tuple


class GameRunError(RuntimeError):
    """Raised when a headless game cannot make safe forward progress."""


def create_headless_game(
    map_num=2,
    num_players=3,
    seed=124,
    use_mission_cards=False,
    use_emperors_favour=False,
    bonus_marker_supply=None,
):
    game = Game(
        map_num=map_num,
        num_players=num_players,
        load_models=False,
        seed=seed,
        interactive_errors=False,
        use_mission_cards=use_mission_cards,
        use_emperors_favour=use_emperors_favour,
        bonus_marker_supply=bonus_marker_supply,
    )
    validate_game(game)
    return game


def legal_action_indices(game):
    mask = masking_out_invalid_actions(game)
    if mask.numel() != 619:
        raise GameRunError(f"Expected a 619-entry action mask, got {mask.numel()}")
    return tuple(int(index) for index in mask.nonzero().flatten().tolist())


def _post_context_for_action(game, action_index):
    post_index = action_index % 121
    current_index = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            if current_index == post_index:
                return route, post
            current_index += 1
    return None, None


def select_progress_action(game, legal_actions, policy_rng):
    """Choose a legal action with a bias toward reaching a terminal state."""
    if 618 in legal_actions:
        return 618

    route_actions = [action for action in legal_actions if 242 <= action <= 521]
    if route_actions:
        def route_index_for_action(action):
            if action < 282:
                return action - 242
            if action < 362:
                return (action - 282) // 2
            return (action - 362) // 4

        marker_route_actions = [
            action
            for action in route_actions
            if game.selected_map.routes[route_index_for_action(action)].bonus_marker
        ]
        if marker_route_actions:
            return policy_rng.choice(marker_route_actions)
        return policy_rng.choice(route_actions)

    income_actions = [action for action in legal_actions if 522 <= action <= 526]
    personal_supply = (
        game.current_player.personal_supply_squares
        + game.current_player.personal_supply_circles
    )
    if income_actions and personal_supply <= 1:
        return policy_rng.choice(income_actions)

    post_actions = [action for action in legal_actions if 0 <= action <= 241]
    pending_post_workflow = game.waiting_for_displaced_player or any(
        value for name, value in vars(game).items() if name.startswith("waiting_for_bm_")
    )
    progressing_post_actions = []
    post_action_scores = {}
    for action in post_actions:
        route, post = _post_context_for_action(game, action)
        if pending_post_workflow or (post is not None and post.owner is not game.current_player):
            progressing_post_actions.append(action)
            post_action_scores[action] = sum(
                route_post.owner is game.current_player for route_post in route.posts
            ) + (100 if route.bonus_marker is not None else 0)
    if progressing_post_actions:
        best_score = max(post_action_scores.values())
        best_actions = [
            action
            for action in progressing_post_actions
            if post_action_scores[action] == best_score
        ]
        return policy_rng.choice(best_actions)

    if income_actions:
        return policy_rng.choice(income_actions)

    if post_actions:
        return policy_rng.choice(post_actions)

    return policy_rng.choice(legal_actions)


def run_game(
    map_num=2,
    num_players=3,
    seed=124,
    max_actions=10_000,
    use_mission_cards=False,
    use_emperors_favour=False,
    bonus_marker_supply=None,
):
    """Run a deterministic legal-action baseline until terminal or a safety limit."""
    game = create_headless_game(
        map_num,
        num_players,
        seed,
        use_mission_cards=use_mission_cards,
        use_emperors_favour=use_emperors_favour,
        bonus_marker_supply=bonus_marker_supply,
    )
    policy_rng = random.Random(seed)
    action_trace = []

    for _ in range(max_actions):
        if game.game_end:
            return GameRunResult(
                map_num=map_num,
                num_players=num_players,
                seed=seed,
                action_count=len(action_trace),
                terminal_reason="game_end",
                final_scores=tuple(player.final_score for player in game.players),
                action_trace=tuple(action_trace),
            )

        legal_actions = legal_action_indices(game)
        if not legal_actions:
            raise GameRunError(
                f"No legal actions after {len(action_trace)} actions; "
                f"player={game.current_player_index}, active_player={game.active_player}"
            )

        action_index = select_progress_action(game, legal_actions, policy_rng)
        action_trace.append(action_index)
        game.apply_action(action_index)
        validate_game(game)

    raise GameRunError(
        f"Game did not finish within {max_actions} actions "
        f"(map={map_num}, players={num_players}, seed={seed})"
    )
