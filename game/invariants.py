class GameInvariantError(AssertionError):
    """Raised when the mutable game graph enters an internally inconsistent state."""


def _require(condition, message):
    if not condition:
        raise GameInvariantError(message)


def validate_game(game):
    """Validate inexpensive invariants that must hold after every complete action."""
    _require(3 <= len(game.players) <= 5, "game must contain between 3 and 5 players")
    _require(
        0 <= game.current_player_index < len(game.players),
        f"invalid current_player_index: {game.current_player_index}",
    )
    _require(
        game.current_player is game.players[game.current_player_index],
        "current_player does not match current_player_index",
    )
    _require(
        0 <= game.active_player < len(game.players),
        f"invalid active_player: {game.active_player}",
    )
    _require(game.turn_number >= 1, f"invalid turn_number: {game.turn_number}")
    _require(game.round_number >= 1, f"invalid round_number: {game.round_number}")
    # Accessing turn_phase also rejects contradictory pending workflows.
    game.turn_phase

    for player in game.players:
        counts = {
            "actions_remaining": player.actions_remaining,
            "general_stock_squares": player.general_stock_squares,
            "general_stock_circles": player.general_stock_circles,
            "personal_supply_squares": player.personal_supply_squares,
            "personal_supply_circles": player.personal_supply_circles,
            "pieces_to_pickup": player.pieces_to_pickup,
            "pieces_to_place": player.pieces_to_place,
        }
        for name, value in counts.items():
            _require(value >= 0, f"player {player.order} has negative {name}: {value}")

    known_players = set(game.players)
    for route in game.selected_map.routes:
        for post in route.posts:
            _require(
                (post.owner is None) == (post.owner_piece_shape is None),
                "post owner and owner_piece_shape disagree",
            )
            _require(
                post.owner is None or post.owner in known_players,
                "post owner is not a player in this game",
            )

    for city in game.selected_map.cities:
        for office in city.offices:
            _require(
                office.controller is None or office.controller in known_players,
                f"office in {city.name} has an unknown controller",
            )

    if game.waiting_for_displaced_player:
        _require(game.displaced_player.player in known_players, "missing displaced player")
        _require(
            game.displaced_player.displaced_shape in ("square", "circle"),
            "invalid displaced piece shape",
        )
        _require(
            game.original_route_of_displacement is not None,
            "displacement is missing its original route",
        )
    else:
        _require(game.displaced_player.player is None, "stale displaced player state")

    if game.game_end:
        for player in game.players:
            _require(player.final_score >= player.score, "terminal final score is below score")

    return True
