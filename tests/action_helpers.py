import json
from pathlib import Path
import tempfile

import torch

from game.persistence import load_game, save_game


EVALUATION_DIRECTORY = Path("training_data/generated/evaluation")


def evaluation_state(name):
    manifest = json.loads((EVALUATION_DIRECTORY / "manifest.json").read_text(encoding="utf-8"))
    entry = next(item for item in manifest if item["name"] == name)
    return EVALUATION_DIRECTORY / Path(entry["save_file"].replace("\\", "/"))


_TEST_STATE_DIRECTORY = tempfile.TemporaryDirectory()


def self_play_test_state():
    path = Path(_TEST_STATE_DIRECTORY.name) / "self-play-terminal.hansa"
    if path.exists():
        return path
    game = load_game(evaluation_state("map1_3p_near_score"))
    player = game.current_player
    player.score = 19
    player.used_bonus_markers.extend(player.bonus_markers)
    player.bonus_markers.clear()
    route = next(
        route
        for route in game.selected_map.routes
        if len(route.posts) == 3
        and not any(post.is_owned() or post.required_shape for post in route.posts)
        and route.bonus_marker is None
        and route.permanent_bonus_marker is None
    )
    for post in route.posts:
        shape = "square"
        supply_field = f"personal_supply_{shape}s"
        stock_field = f"general_stock_{shape}s"
        source = supply_field if getattr(player, supply_field) else stock_field
        setattr(player, source, getattr(player, source) - 1)
        post.owner = player
        post.owner_piece_shape = shape
    used_routes = {route}
    for shape in ("square", "circle"):
        supply_field = f"personal_supply_{shape}s"
        stock_field = f"general_stock_{shape}s"
        pieces = getattr(player, supply_field) + getattr(player, stock_field)
        candidates = [
            (
                route,
                next(
                    post
                    for post in route.posts
                    if post.owner is None and post.can_be_claimed_by(shape)
                ),
            )
            for route in game.selected_map.routes
            if route not in used_routes
            and any(post.owner is None and post.can_be_claimed_by(shape) for post in route.posts)
        ]
        if len(candidates) < pieces:
            raise RuntimeError("Evaluation state cannot supply the self-play test fixture")
        for route, post in candidates[:pieces]:
            post.owner = player
            post.owner_piece_shape = shape
            used_routes.add(route)
        setattr(player, supply_field, 0)
        setattr(player, stock_field, 0)
    return save_game(game, path)


def legal_action_mask(game):
    """Return a tensor mask for assertions that use PyTorch indexing helpers."""
    return torch.tensor(game.ai_action_mask(), dtype=torch.uint8)
