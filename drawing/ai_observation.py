"""Compatibility access to the engine-owned, player-visible observation."""


def public_game_state(encoder, game, observer):
    """Return the same headless features used by AI players."""
    observer_index = game.players.index(observer)
    return encoder.get_game_state(game, observer_index=observer_index)
