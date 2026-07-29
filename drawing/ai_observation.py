"""Player-perspective filtering for AI decisions made through the GUI."""

from __future__ import annotations


def public_game_state(board_data, game, observer):
    """Return an observation with private and face-down information concealed."""
    state = board_data.get_game_state(game).clone()

    # Game tensor layout: 17 state flags, 3 regional privileges, 4 prestige
    # owners, then 12 face-down bonus-marker supply slots.
    supply_start = 24
    state[supply_start : supply_start + 12] = 0

    player_start = (
        board_data.game_tensor_size + board_data.city_tensor_size + board_data.route_tensor_size
    )
    player_width = 55
    mission_offset = 20
    used_marker_offset = 35

    for player_index, player in enumerate(game.players):
        start = player_start + player_index * player_width
        state[start + mission_offset : start + mission_offset + 3] = 0
        if player is not observer:
            state[start + used_marker_offset : start + used_marker_offset + 12] = 0
            state[start + used_marker_offset] = len(player.used_bonus_markers)

    if game.use_mission_cards and observer.mission_card:
        observer_start = player_start + game.players.index(observer) * player_width
        encoded_cities = []
        for city_name in observer.mission_card:
            city = next(city for city in game.selected_map.cities if city.name == city_name)
            city_number, _color = board_data.assign_city_name_and_color_mapping(game, city)
            encoded_cities.append(city_number)
        state[observer_start + mission_offset : observer_start + mission_offset + 3] = (
            state.new_tensor(encoded_cities)
        )

    return state
