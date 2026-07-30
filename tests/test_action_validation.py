import contextlib
import io
import pickle
import random
import unittest

from game.action_validation import state_fingerprint, validate_action_state
from game.game_actions import refresh_displacement_targets
from game.game_runner import (
    create_headless_game,
    legal_action_indices,
    replay_game,
    select_progress_action,
)
from game.structured_actions import PieceShape, PostInteraction, RouteInteraction
from map_data.map_attributes import BonusMarker


def occupy_route(player, route):
    for post in route.posts:
        post.claim(player, "square")
        player.personal_supply_squares -= 1


class ExhaustiveActionValidationTests(unittest.TestCase):
    def validate_quietly(self, game):
        with contextlib.redirect_stdout(io.StringIO()):
            return validate_action_state(game)

    def test_normal_actions_income_mask_codec_and_execution_agree(self):
        game = create_headless_game(2, 3, seed=124)
        result = self.validate_quietly(game)
        self.assertEqual(result.legal_action_count, result.enabled_index_count)

    def test_route_office_upgrade_and_prestige_outcomes_execute(self):
        for special_prestige in (False, True):
            with self.subTest(special_prestige=special_prestige):
                game = create_headless_game(2, 3, seed=124)
                player = game.current_player
                route = next(
                    route
                    for route in game.selected_map.routes
                    if (
                        any(
                            "SpecialPrestigePoints" in city.upgrade_city_type
                            for city in route.cities
                        )
                        if special_prestige
                        else any(
                            city.upgrade_city_type
                            and "SpecialPrestigePoints" not in city.upgrade_city_type
                            for city in route.cities
                        )
                    )
                    and all(post.required_shape != "circle" for post in route.posts)
                )
                shapes = ["square"] * len(route.posts)
                if special_prestige:
                    shapes[0] = "circle"
                for post, shape in zip(route.posts, shapes):
                    post.claim(player, shape)
                    if shape == "circle":
                        player.personal_supply_circles -= 1
                    else:
                        player.personal_supply_squares -= 1

                result = self.validate_quietly(game)
                self.assertEqual(result.legal_action_count, result.enabled_index_count)

    def test_displacement_optional_supply_finish_and_destinations_execute(self):
        game = create_headless_game(2, 3, seed=124)
        opponent = game.players[1]
        route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Malmo", "Visby"}
        )
        opponent.perform_upgrade("Book")
        origin = route.posts[0]
        origin.claim(opponent, "circle")
        opponent.personal_supply_circles -= 1
        origin.owner = None
        origin.owner_piece_shape = None
        opponent.personal_supply_squares += opponent.general_stock_squares - 1
        opponent.personal_supply_circles += opponent.general_stock_circles - 1
        opponent.general_stock_squares = 1
        opponent.general_stock_circles = 1
        game.original_route_of_displacement = route
        game.waiting_for_displaced_player = True
        game.displaced_player.populate_displaced_player(game, opponent, "circle")
        refresh_displacement_targets(game)

        self.validate_quietly(game)
        mandatory_destination = next(
            action
            for action in game.get_legal_actions()
            if isinstance(action, PostInteraction) and action.shape is PieceShape.MERCHANT
        )
        game.apply_structured_action(mandatory_destination)
        self.validate_quietly(game)

    def test_terminal_state_exposes_no_actions(self):
        game = create_headless_game(2, 3, seed=124)
        game.game_end = True
        result = self.validate_quietly(game)
        self.assertEqual(result.legal_action_count, 0)

    def test_bonus_marker_tile_city_ability_and_end_turn_phases(self):
        exchange = create_headless_game(1, 3, seed=124)
        exchange_player, exchange_target = exchange.players[:2]
        exchange_player.bonus_markers = [BonusMarker("ExchangeBonusMarker", owner=exchange_player)]
        exchange_target.used_bonus_markers = [BonusMarker("Move3", owner=exchange_target)]
        exchange.waiting_for_bm_exchange_bm = True
        exchange.pending_exchange_marker = exchange_player.bonus_markers[0]
        self.validate_quietly(exchange)
        exchange.exchange_target_player = exchange_target
        self.validate_quietly(exchange)

        ability = create_headless_game(1, 3, seed=124)
        ability.waiting_for_bm_upgrade_ability = True
        self.validate_quietly(ability)

        payment = create_headless_game(1, 3, seed=124, use_emperors_favour=True)
        owner = payment.current_player
        owner.bonus_markers = [
            BonusMarker("Move3", owner=owner),
            BonusMarker("3Actions", owner=owner),
        ]
        payment.waiting_for_buy_tile_with_bm = True
        payment.first_bm_to_spend_on_tile = owner.bonus_markers[0]
        payment.tile_to_buy = payment.tile_pool[0]
        self.validate_quietly(payment)

        tile_selection = create_headless_game(1, 3, seed=124, use_emperors_favour=True)
        tile_selection.current_player.bonus_markers = [
            BonusMarker("Move3", owner=tile_selection.current_player),
            BonusMarker("3Actions", owner=tile_selection.current_player),
        ]
        tile_selection.waiting_for_buy_tile_with_bm = True
        self.validate_quietly(tile_selection)

        swap = create_headless_game(1, 3, seed=124)
        city = next(
            city
            for city in swap.selected_map.cities
            if any(left.shape != right.shape for left, right in zip(city.offices, city.offices[1:]))
        )
        pair_start = next(
            index
            for index, (left, right) in enumerate(zip(city.offices, city.offices[1:]))
            if left.shape != right.shape
        )
        first, second = swap.players[:2]
        offices = city.offices[pair_start : pair_start + 2]
        offices[0].controller = first
        offices[1].controller = second
        offices[0].owner_piece_shape = offices[0].shape
        offices[1].owner_piece_shape = offices[1].shape
        for player, office in zip((first, second), offices):
            attribute = (
                "personal_supply_circles" if office.shape == "circle" else "personal_supply_squares"
            )
            setattr(player, attribute, getattr(player, attribute) - 1)
        swap.waiting_for_bm_swap_office = True
        self.validate_quietly(swap)

        end_turn = create_headless_game(1, 3, seed=124)
        end_turn.current_player.actions_remaining = 0
        end_turn.current_player.bonus_markers = [
            BonusMarker("3Actions", owner=end_turn.current_player)
        ]
        self.validate_quietly(end_turn)

    def test_remaining_workflow_phases_are_exhaustively_validated(self):
        move = create_headless_game(2, 3, seed=124)
        player, first_owner, second_owner = move.players
        first, second = move.selected_map.routes[0].posts[:2]
        first.claim(first_owner, "square")
        second.claim(second_owner, "circle")
        first_owner.personal_supply_squares -= 1
        second_owner.personal_supply_circles -= 1
        move.waiting_for_bm_move3 = True
        player.pieces_to_pickup = 3
        self.validate_quietly(move)

        income_favour = create_headless_game(1, 3, seed=124)
        owner, other = income_favour.players[:2]
        income_favour.OneIncomeIfOthersIncomeOwner = owner
        owner.personal_supply_circles -= 1
        owner.general_stock_circles += 1
        income_favour.begin_income_favour_response(other)
        self.validate_quietly(income_favour)

        tribute = create_headless_game(1, 3, seed=124)
        tribute.pending_tribute_income_owners = [tribute.current_player]
        self.validate_quietly(tribute)

        adjacent = create_headless_game(1, 3, seed=124)
        adjacent.current_player.bonus_markers = [
            BonusMarker("PlaceAdjacent", owner=adjacent.current_player)
        ]
        adjacent_route = adjacent.selected_map.routes[0]
        occupy_route(adjacent.current_player, adjacent_route)
        office = adjacent_route.cities[0].offices[0]
        office.controller = adjacent.players[1]
        office.owner_piece_shape = office.shape
        attribute = (
            "personal_supply_circles" if office.shape == "circle" else "personal_supply_squares"
        )
        setattr(adjacent.players[1], attribute, getattr(adjacent.players[1], attribute) - 1)
        adjacent.waiting_for_bm_place_adjacent = True
        self.validate_quietly(adjacent)

        permanent = create_headless_game(2, 3, seed=124)
        permanent_route = next(
            route
            for route in permanent.selected_map.routes
            if route.has_permanent_bm_type == "Place2TradesmenFromRoute"
        )
        occupy_route(permanent.current_player, permanent_route)
        permanent.apply_structured_action(
            RouteInteraction(permanent.selected_map.routes.index(permanent_route), 0)
        )
        self.validate_quietly(permanent)

        replacement = create_headless_game(2, 3, seed=124)
        replacement.current_player.forfeit_remaining_actions()
        replacement.current_player.ending_turn = True
        replacement.replace_bonus_marker = 1
        self.validate_quietly(replacement)

    def test_displacement_personal_supply_and_board_fallback_states(self):
        personal = create_headless_game(2, 3, seed=124)
        opponent = personal.players[1]
        displaced = personal.selected_map.routes[0].posts[0]
        displaced.claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        opponent.personal_supply_squares += opponent.general_stock_squares
        opponent.personal_supply_circles += opponent.general_stock_circles
        opponent.general_stock_squares = 0
        opponent.general_stock_circles = 0
        personal.apply_structured_action(PostInteraction(0, PieceShape.TRADER))
        self.validate_quietly(personal)

        board = create_headless_game(2, 3, seed=124)
        opponent = board.players[1]
        displaced = board.selected_map.routes[0].posts[0]
        fallback_route = board.selected_map.routes[-1]
        fallback = fallback_route.posts[0]
        displaced.claim(opponent, "square")
        fallback.claim(opponent, "square")
        opponent.personal_supply_squares -= 2
        remaining_shapes = ["square"] * (
            opponent.general_stock_squares + opponent.personal_supply_squares
        ) + ["circle"] * (opponent.general_stock_circles + opponent.personal_supply_circles)
        offices = [
            office
            for city in board.selected_map.cities
            for office in city.offices
            if office.controller is None
        ]
        for shape, office in zip(remaining_shapes, offices):
            office.controller = opponent
            office.owner_piece_shape = shape
        opponent.general_stock_squares = 0
        opponent.general_stock_circles = 0
        opponent.personal_supply_squares = 0
        opponent.personal_supply_circles = 0
        board.apply_structured_action(PostInteraction(0, PieceShape.TRADER))
        mandatory = next(
            action
            for action in board.get_legal_actions()
            if isinstance(action, PostInteraction) and action.post_slot != 0
        )
        board.apply_structured_action(mandatory)
        self.validate_quietly(board)

    def test_save_load_restored_state_preserves_legal_interactions(self):
        game = create_headless_game(2, 3, seed=124, use_emperors_favour=True)
        office = next(office for city in game.selected_map.cities for office in city.offices)
        office.controller = game.current_player
        office.owner_piece_shape = "circle" if office.shape == "square" else "square"
        supply_attribute = (
            "personal_supply_circles"
            if office.owner_piece_shape == "circle"
            else "personal_supply_squares"
        )
        setattr(
            game.current_player,
            supply_attribute,
            getattr(game.current_player, supply_attribute) - 1,
        )
        game.tile_pool.pop()
        restored = pickle.loads(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))

        self.assertEqual(game.get_legal_actions(), restored.get_legal_actions())
        restored_office = next(
            office for city in restored.selected_map.cities for office in city.offices
        )
        self.assertEqual(restored_office.owner_piece_shape, office.owner_piece_shape)
        self.assertEqual(len(restored.tile_pool), len(game.tile_pool))
        self.validate_quietly(restored)

    def test_replay_restored_state_preserves_state_and_actions(self):
        original = create_headless_game(2, 3, seed=124)
        policy_rng = random.Random(124)
        trace = []
        for _ in range(4):
            legal = legal_action_indices(original)
            action = select_progress_action(original, legal, policy_rng)
            trace.append(action)
            original.apply_ai_action(action)

        replay = replay_game(trace, map_num=2, num_players=3, seed=124)

        self.assertEqual(state_fingerprint(original), state_fingerprint(replay))
        self.validate_quietly(replay)


if __name__ == "__main__":
    unittest.main()
