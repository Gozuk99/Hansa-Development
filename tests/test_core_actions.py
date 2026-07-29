import contextlib
import io
import unittest

from ai.action_options import InvalidActionError
from game.game_actions import refresh_displacement_targets
from game.game_runner import create_headless_game
from game.turn_state import TurnPhase


MAX_POSTS = 121
MAX_ROUTES = 40


def post_index(game, target_post, shape="square"):
    index = 0
    for route in game.selected_map.routes:
        for post in route.posts:
            if post is target_post:
                return index + (MAX_POSTS if shape == "circle" else 0)
            index += 1
    raise AssertionError("Post not found")


def route_points_index(game, route):
    return 242 + game.selected_map.routes.index(route)


def route_office_index(game, route, city):
    route_index = game.selected_map.routes.index(route)
    city_index = route.cities.index(city)
    return 242 + MAX_ROUTES + route_index * 2 + city_index


def route_upgrade_index(game, route, city, upgrade_index=0):
    route_index = game.selected_map.routes.index(route)
    city_index = route.cities.index(city)
    return 242 + MAX_ROUTES * 3 + route_index * 4 + city_index * 2 + upgrade_index


def occupy_route(player, route, shapes=None):
    shapes = shapes or ["square"] * len(route.posts)
    for post, shape in zip(route.posts, shapes):
        post.claim(player, shape)
        if shape == "square":
            player.personal_supply_squares -= 1
        else:
            player.personal_supply_circles -= 1


class CoreActionTests(unittest.TestCase):
    def apply(self, game, action_index):
        with contextlib.redirect_stdout(io.StringIO()):
            game.apply_action(action_index)

    def city_path(self, game, start_name, end_name):
        start = next(city for city in game.selected_map.cities if city.name == start_name)
        end = next(city for city in game.selected_map.cities if city.name == end_name)
        queue = [(start, [start])]
        visited = {start}
        while queue:
            city, path = queue.pop(0)
            if city is end:
                return path
            for route in city.routes:
                for neighbor in route.cities:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        raise AssertionError("No city path found")

    def test_income_obeys_every_bank_capacity_and_conserves_pieces(self):
        cases = (
            (3, 3, 0, 522),
            (4, 3, 1, 523),
            (7, 4, 3, 525),
            (50, 4, 4, 526),
        )
        for bank, squares, circles, action_index in cases:
            with self.subTest(bank=bank):
                game = create_headless_game(2, 3, seed=124)
                player = game.current_player
                player.bank = bank
                player.general_stock_squares = squares
                player.general_stock_circles = circles
                player.personal_supply_squares = 0
                player.personal_supply_circles = 0
                total_before = squares + circles

                self.apply(game, action_index)

                self.assertEqual(
                    player.personal_supply_squares + player.personal_supply_circles,
                    total_before,
                )
                self.assertEqual(
                    player.general_stock_squares + player.general_stock_circles,
                    0,
                )
                self.assertEqual(player.actions_remaining, 1)

    def test_income_mutation_rejects_over_capacity_and_unavailable_shapes(self):
        player = create_headless_game(2, 3, seed=124).current_player
        player.general_stock_circles = 1
        with self.assertRaises(ValueError):
            player.income_action(3, 1)
        with self.assertRaises(ValueError):
            player.income_action(0, 2)
        with self.assertRaises(ValueError):
            player.income_action(0, 0)

    def test_income_mask_rejects_zero_piece_composition(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        player.general_stock_squares = 0
        player.general_stock_circles = 1

        mask = game.legal_action_mask()

        self.assertEqual(mask[522].item(), 0)
        self.assertEqual(mask[523].item(), 1)

    def test_place_tradesman_costs_one_action_and_one_supply_piece(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        route = game.selected_map.routes[0]
        post = route.posts[0]
        supply_before = player.personal_supply_squares

        self.apply(game, post_index(game, post))

        self.assertIs(post.owner, player)
        self.assertEqual(post.owner_piece_shape, "square")
        self.assertEqual(player.personal_supply_squares, supply_before - 1)
        self.assertEqual(player.actions_remaining, 1)

    def test_required_circle_post_rejects_trader_and_accepts_merchant(self):
        game = create_headless_game(2, 3, seed=124)
        route = next(route for route in game.selected_map.routes if route.required_circles)
        post = next(post for post in route.posts if post.required_shape == "circle")
        mask = game.legal_action_mask()
        self.assertEqual(mask[post_index(game, post, "square")].item(), 0)
        self.assertEqual(mask[post_index(game, post, "circle")].item(), 1)

        self.apply(game, post_index(game, post, "circle"))
        self.assertEqual(post.owner_piece_shape, "circle")

    def test_displacing_trader_pays_one_extra_and_may_decline_extra_piece(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        occupied = route.posts[0]
        occupied.claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        actor_supply_before = actor.personal_supply_squares
        actor_stock_before = actor.general_stock_squares

        self.apply(game, post_index(game, occupied))

        self.assertEqual(game.turn_phase, TurnPhase.DISPLACEMENT)
        self.assertIs(occupied.owner, actor)
        self.assertEqual(actor.personal_supply_squares, actor_supply_before - 2)
        self.assertEqual(actor.general_stock_squares, actor_stock_before + 1)

        displacement_mask = game.legal_action_mask()
        displaced_post_index = next(
            index
            for index in displacement_mask[:242].nonzero(as_tuple=True)[0].tolist()
            if index < 121
        )
        self.apply(game, displaced_post_index)
        self.assertEqual(game.legal_action_mask()[618].item(), 1)
        self.apply(game, 618)

        self.assertFalse(game.waiting_for_displaced_player)
        self.assertEqual(actor.actions_remaining, 1)
        opponent_posts = sum(
            post.owner is opponent
            for candidate_route in game.selected_map.routes
            for post in candidate_route.posts
        )
        self.assertEqual(opponent_posts, 1)

    def test_displacing_merchant_costs_two_extra_pieces(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        occupied = route.posts[0]
        occupied.claim(opponent, "circle")
        opponent.personal_supply_circles -= 1
        actor_supply_before = actor.personal_supply_squares
        actor_stock_before = actor.general_stock_squares

        self.apply(game, post_index(game, occupied))

        self.assertEqual(actor.personal_supply_squares, actor_supply_before - 3)
        self.assertEqual(actor.general_stock_squares, actor_stock_before + 2)
        self.assertEqual(game.displaced_player.total_pieces_to_place, 3)

    def test_displaced_extra_piece_comes_from_general_stock_and_all_may_be_placed(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        occupied = route.posts[0]
        occupied.claim(opponent, "square")
        opponent.personal_supply_squares -= 1
        stock_before = opponent.general_stock_squares
        self.apply(game, post_index(game, occupied))

        for _ in range(2):
            mask = game.legal_action_mask()
            target = next(index for index in mask[:121].nonzero(as_tuple=True)[0].tolist())
            self.apply(game, target)

        self.assertFalse(game.waiting_for_displaced_player)
        self.assertEqual(opponent.general_stock_squares, stock_before - 1)
        self.assertEqual(
            sum(
                post.owner is opponent
                for candidate_route in game.selected_map.routes
                for post in candidate_route.posts
            ),
            2,
        )

    def test_displaced_extra_piece_falls_back_to_personal_supply_only_when_stock_empty(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        occupied = route.posts[0]
        occupied.claim(opponent, "square")
        opponent.general_stock_squares = 0
        opponent.general_stock_circles = 0
        personal_before = opponent.personal_supply_squares
        self.apply(game, post_index(game, occupied))

        for _ in range(2):
            target = next(
                index for index in game.legal_action_mask()[:121].nonzero(as_tuple=True)[0].tolist()
            )
            self.apply(game, target)

        self.assertEqual(opponent.personal_supply_squares, personal_before - 1)

    def test_displacement_targets_nearest_adjacent_routes_first(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        route.posts[0].claim(opponent, "square")
        self.apply(game, post_index(game, route.posts[0]))

        directly_adjacent = {
            adjacent for city in route.cities for adjacent in city.routes if adjacent is not route
        }
        legal_post_numbers = {
            index % MAX_POSTS
            for index in game.legal_action_mask()[:242].nonzero(as_tuple=True)[0].tolist()
        }
        actual_routes = set()
        running_index = 0
        for candidate_route in game.selected_map.routes:
            for _post in candidate_route.posts:
                if running_index in legal_post_numbers:
                    actual_routes.add(candidate_route)
                running_index += 1
        self.assertTrue(actual_routes)
        self.assertTrue(actual_routes.issubset(directly_adjacent))

    def test_displaced_extra_piece_falls_back_to_piece_already_on_board(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        original_route = game.selected_map.routes[0]
        displaced_post = original_route.posts[0]
        board_fallback_post = game.selected_map.routes[-1].posts[0]
        displaced_post.claim(opponent, "square")
        board_fallback_post.claim(opponent, "square")
        opponent.general_stock_squares = 0
        opponent.general_stock_circles = 0
        opponent.personal_supply_squares = 0
        opponent.personal_supply_circles = 0
        self.apply(game, post_index(game, displaced_post))

        first_target = next(
            index for index in game.legal_action_mask()[:121].nonzero(as_tuple=True)[0].tolist()
        )
        self.apply(game, first_target)
        self.assertEqual(opponent.pieces_to_pickup, 1)
        self.assertEqual(
            game.legal_action_mask()[post_index(game, board_fallback_post)].item(),
            1,
        )
        self.apply(game, post_index(game, board_fallback_post))
        final_target = next(
            index for index in game.legal_action_mask()[:121].nonzero(as_tuple=True)[0].tolist()
        )
        self.apply(game, final_target)

        self.assertFalse(game.waiting_for_displaced_player)
        self.assertEqual(
            sum(
                post.owner is opponent for route in game.selected_map.routes for post in route.posts
            ),
            2,
        )

    def test_empty_supply_relocated_merchant_stops_at_nearest_circle_post(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        original_route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Malmo", "Visby"}
        )
        adjacent_routes = {
            route
            for city in original_route.cities
            for route in city.routes
            if route is not original_route
        }

        # Leave only Merchant posts open at distance one so the displaced
        # Tradesman must continue outward. These blockers belong to the actor,
        # so the empty-supply fallback cannot select them for relocation.
        for route in adjacent_routes:
            for post in route.posts:
                if post.required_shape is None:
                    post.claim(actor, "square")

        displaced_post = next(post for post in original_route.posts if post.required_shape is None)
        displaced_post.claim(opponent, "square")
        board_circle = next(
            post
            for route in reversed(game.selected_map.routes)
            for post in route.posts
            if route is not original_route
            and route not in adjacent_routes
            and post.required_shape is None
            and not post.is_owned()
        )
        board_circle.claim(opponent, "circle")
        opponent.general_stock_squares = 0
        opponent.general_stock_circles = 0
        opponent.personal_supply_squares = 0
        opponent.personal_supply_circles = 0

        self.apply(game, post_index(game, displaced_post))
        initial_mask = game.legal_action_mask()
        self.assertFalse(
            any(
                initial_mask[post_index(game, post)].item()
                for route in adjacent_routes
                for post in route.posts
                if post.required_shape == "circle"
            )
        )
        outward_square = next(
            index for index in initial_mask[:MAX_POSTS].nonzero(as_tuple=True)[0].tolist()
        )
        self.apply(game, outward_square)

        self.apply(game, post_index(game, board_circle))
        relocated_mask = game.legal_action_mask()
        adjacent_circle = next(
            post
            for route in adjacent_routes
            for post in route.posts
            if post.required_shape == "circle" and not post.is_owned()
        )
        self.assertEqual(
            relocated_mask[post_index(game, adjacent_circle, "circle")].item(),
            1,
        )
        farther_circle_targets = [
            index
            for index in relocated_mask[MAX_POSTS : MAX_POSTS * 2]
            .nonzero(as_tuple=True)[0]
            .tolist()
            if index
            not in {post_index(game, post) for route in adjacent_routes for post in route.posts}
        ]
        self.assertEqual(farther_circle_targets, [])

        self.apply(game, post_index(game, adjacent_circle, "circle"))
        self.assertFalse(game.waiting_for_displaced_player)
        self.assertIs(adjacent_circle.owner, opponent)
        self.assertEqual(adjacent_circle.owner_piece_shape, "circle")

    def test_displaced_merchant_extra_stock_pieces_must_be_merchants(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        original_route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Malmo", "Visby"}
        )
        displaced_post = next(post for post in original_route.posts if post.required_shape is None)
        displaced_post.claim(opponent, "circle")
        opponent.general_stock_circles = 2
        opponent.general_stock_squares = 2

        self.apply(game, post_index(game, displaced_post))
        first_circle = next(
            index
            for index in game.legal_action_mask()[MAX_POSTS : MAX_POSTS * 2]
            .nonzero(as_tuple=True)[0]
            .tolist()
        )
        self.apply(game, MAX_POSTS + first_circle)

        stock_squares_before = opponent.general_stock_squares
        stock_circles_before = opponent.general_stock_circles
        extra_mask = game.legal_action_mask()
        adjacent_routes = {
            route
            for city in original_route.cities
            for route in city.routes
            if route is not original_route
        }
        adjacent_post_indices = {
            post_index(game, post) for route in adjacent_routes for post in route.posts
        }
        legal_circle_indices = set(
            extra_mask[MAX_POSTS : MAX_POSTS * 2].nonzero(as_tuple=True)[0].tolist()
        )
        self.assertEqual(extra_mask[:MAX_POSTS].count_nonzero().item(), 0)
        self.assertTrue(legal_circle_indices)
        self.assertTrue(legal_circle_indices.issubset(adjacent_post_indices))
        self.assertEqual(extra_mask[618].item(), 1)

        second_circle = next(iter(legal_circle_indices))
        self.apply(game, MAX_POSTS + second_circle)
        self.assertEqual(opponent.general_stock_squares, stock_squares_before)
        self.assertEqual(
            opponent.general_stock_circles,
            stock_circles_before - 1,
        )

    def test_britannia_board_fallback_keeps_shape_and_country_search_rules(self):
        game = create_headless_game(3, 4, seed=124)
        opponent = game.players[1]
        original_route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Carlisle", "IsleOfMan"}
        )
        legal_adjacent_routes = {
            route
            for city in original_route.cities
            for route in city.routes
            if route is not original_route and route.region in ("Scotland", None)
        }
        wales_maritime_route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Conway", "IsleOfMan"}
        )
        nearest_circle_route = next(
            route
            for route in legal_adjacent_routes
            if {city.name for city in route.cities} == {"Carlisle", "Chester"}
        )

        # Force the nearest valid distance to contain only Merchant spaces.
        for route in legal_adjacent_routes:
            for post in route.posts:
                if post.required_shape is None:
                    post.claim(game.current_player, "square")

        opponent.general_stock_squares = 0
        opponent.general_stock_circles = 0
        opponent.personal_supply_squares = 0
        opponent.personal_supply_circles = 0
        opponent.holding_pieces = [("circle", opponent, "Scotland")]
        opponent.pieces_to_pickup = 0
        game.original_route_of_displacement = original_route
        game.waiting_for_displaced_player = True
        game.displaced_player.populate_displaced_player(game, opponent, "square")
        game.displaced_player.played_displaced_shape = True

        targets = refresh_displacement_targets(game)
        mask = game.legal_action_mask()

        self.assertTrue(targets)
        self.assertTrue(all(post in nearest_circle_route.posts for post in targets))
        self.assertTrue(all(post.required_shape == "circle" for post in targets))
        self.assertFalse(any(post in wales_maritime_route.posts for post in targets))
        self.assertEqual(mask[:MAX_POSTS].count_nonzero().item(), 0)
        self.assertEqual(
            {index for index in mask[MAX_POSTS : MAX_POSTS * 2].nonzero(as_tuple=True)[0].tolist()},
            {post_index(game, post) for post in targets},
        )

    def test_britannia_place_requires_and_consumes_one_regional_permission(self):
        game = create_headless_game(3, 4, seed=124)
        player = game.current_player
        wales_route = next(route for route in game.selected_map.routes if route.region == "Wales")
        target = wales_route.posts[0]
        target_shape = target.required_shape or "square"
        target_index = post_index(game, target, target_shape)
        self.assertTrue(all(post.region == "Wales" for post in wales_route.posts))
        self.assertEqual(game.legal_action_mask()[target_index].item(), 0)

        cardiff = next(city for city in game.selected_map.cities if city.name == "Cardiff")
        cardiff.offices[0].controller = player
        player.refresh_map3_priv_actions(game)
        self.assertEqual(game.legal_action_mask()[target_index].item(), 1)
        self.apply(game, target_index)
        self.assertEqual(player.brown_priv_count, 0)

    def test_london_permission_is_one_shared_wales_or_scotland_use(self):
        game = create_headless_game(3, 4, seed=124)
        player = game.current_player
        wales_route = next(route for route in game.selected_map.routes if route.region == "Wales")
        scotland_route = next(
            route for route in game.selected_map.routes if route.region == "Scotland"
        )
        london = next(city for city in game.selected_map.cities if city.name == "London")
        london.offices[0].controller = player
        player.refresh_map3_priv_actions(game)
        self.assertEqual(player.london_priv_count, 1)

        wales_target = wales_route.posts[0]
        wales_shape = wales_target.required_shape or "square"
        self.apply(game, post_index(game, wales_target, wales_shape))

        self.assertEqual(player.london_priv_count, 0)
        self.assertEqual(
            game.legal_action_mask()[
                post_index(
                    game,
                    scotland_route.posts[0],
                    scotland_route.posts[0].required_shape or "square",
                )
            ].item(),
            0,
        )

    def test_normal_move_can_swap_own_trader_and_merchant(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        route = game.selected_map.routes[0]
        first, second = route.posts[:2]
        first.claim(player, "square")
        second.claim(player, "circle")
        player.personal_supply_squares -= 1
        player.personal_supply_circles -= 1

        self.apply(game, post_index(game, first))
        self.apply(game, post_index(game, second, "circle"))
        self.apply(game, post_index(game, second))
        self.apply(game, post_index(game, first, "circle"))

        self.assertEqual(first.owner_piece_shape, "circle")
        self.assertEqual(second.owner_piece_shape, "square")
        self.assertEqual(player.actions_remaining, 1)
        self.assertEqual(game.turn_phase, TurnPhase.ACTIONS)

    def test_normal_move_never_exposes_opponent_as_pickup_or_target(self):
        game = create_headless_game(2, 3, seed=124)
        player, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        own_post, opponent_post = route.posts[:2]
        own_post.claim(player, "square")
        opponent_post.claim(opponent, "square")

        self.apply(game, post_index(game, own_post))
        mask = game.legal_action_mask()
        self.assertEqual(mask[post_index(game, opponent_post)].item(), 0)
        self.assertEqual(mask[post_index(game, opponent_post, "circle")].item(), 0)

    def test_move_pickups_cannot_exceed_book_value(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        player.book = 2
        route = game.selected_map.routes[0]
        for post in route.posts[:3]:
            post.claim(player, "square")

        self.apply(game, post_index(game, route.posts[0]))
        self.apply(game, post_index(game, route.posts[1]))
        mask = game.legal_action_mask()
        self.assertEqual(mask[post_index(game, route.posts[2])].item(), 0)

    def test_route_points_requires_full_control_and_returns_all_pieces(self):
        game = create_headless_game(2, 3, seed=124)
        player, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        action_index = route_points_index(game, route)
        self.assertEqual(game.legal_action_mask()[action_index].item(), 0)

        occupy_route(player, route)
        route.cities[0].offices[0].controller = player
        route.cities[1].offices[0].controller = opponent
        stock_before = player.general_stock_squares

        self.assertEqual(game.legal_action_mask()[action_index].item(), 1)
        self.apply(game, action_index)

        self.assertEqual(player.score, 1)
        self.assertEqual(opponent.score, 1)
        self.assertTrue(all(post.owner is None for post in route.posts))
        self.assertEqual(player.general_stock_squares, stock_before + len(route.posts))
        self.assertEqual(player.actions_remaining, 1)

    def test_route_bonus_marker_is_taken_and_replacement_drawn_immediately(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        route = next(route for route in game.selected_map.routes if route.bonus_marker)
        marker_type = route.bonus_marker.type
        pool_before = len(game.selected_map.bonus_marker_pool)
        occupy_route(player, route)

        self.apply(game, route_points_index(game, route))

        self.assertIn(marker_type, [marker.type for marker in player.bonus_markers])
        self.assertIsNone(route.bonus_marker)
        self.assertEqual(len(game.selected_map.bonus_marker_pool), pool_before - 1)
        self.assertEqual(len(game.pending_bonus_markers), 1)
        self.assertEqual(game.replace_bonus_marker, 1)

    def test_pending_replacement_marker_obeys_all_three_route_restrictions(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        claimed_route = next(route for route in game.selected_map.routes if route.bonus_marker)
        occupy_route(player, claimed_route)
        self.apply(game, route_points_index(game, claimed_route))
        player.forfeit_remaining_actions()
        player.ending_turn = True

        mask = game.legal_action_mask()
        legal_routes = mask[543:583].nonzero(as_tuple=True)[0].tolist()
        self.assertTrue(legal_routes)
        for route_index in legal_routes:
            route = game.selected_map.routes[route_index]
            self.assertIsNone(route.bonus_marker)
            self.assertIsNone(route.permanent_bonus_marker)
            self.assertFalse(route.has_tradesmen())
            self.assertTrue(route.has_empty_office_in_cities())

        target_index = legal_routes[0]
        marker_type = game.pending_bonus_markers[0]
        self.apply(game, 543 + target_index)
        self.assertEqual(
            game.selected_map.routes[target_index].bonus_marker.type,
            marker_type,
        )
        self.assertEqual(game.replace_bonus_marker, 0)
        self.assertEqual(game.pending_bonus_markers, [])

    def test_empty_marker_supply_ends_only_when_replacement_draw_is_required(self):
        game = create_headless_game(2, 3, seed=124)
        game.selected_map.bonus_marker_pool.clear()
        game.check_for_game_end()
        self.assertFalse(game.game_end)

        route = next(route for route in game.selected_map.routes if route.bonus_marker)
        occupy_route(game.current_player, route)
        self.apply(game, route_points_index(game, route))

        self.assertTrue(game.game_end)
        self.assertFalse(route.has_tradesmen())
        self.assertEqual(game.replace_bonus_marker, 0)

    def test_office_route_claim_uses_leftmost_shape_and_returns_other_pieces(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        city = next(city for city in game.selected_map.cities if city.name == "Groningen")
        route = city.routes[0]
        occupy_route(player, route)
        stock_before = player.general_stock_squares

        self.apply(game, route_office_index(game, route, city))

        self.assertIs(city.offices[0].controller, player)
        self.assertEqual(player.score, 1)
        self.assertEqual(
            player.general_stock_squares,
            stock_before + len(route.posts) - 1,
        )
        self.assertTrue(all(post.owner is None for post in route.posts))

    def test_office_privilege_thresholds_are_enforced_in_route_mask(self):
        game = create_headless_game(2, 3, seed=124)
        player, opponent = game.players[:2]
        city = next(
            city
            for city in game.selected_map.cities
            if len(city.offices) >= 2 and city.offices[1].color == "ORANGE"
        )
        route = city.routes[0]
        city.offices[0].controller = opponent
        occupy_route(player, route)
        action_index = route_office_index(game, route, city)

        self.assertEqual(game.legal_action_mask()[action_index].item(), 0)
        player.privilege = "ORANGE"
        self.assertEqual(game.legal_action_mask()[action_index].item(), 1)

    def test_circle_only_route_cannot_claim_square_office(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if any(city.get_next_open_office_shape() == "square" for city in route.cities)
        )
        city = next(city for city in route.cities if city.get_next_open_office_shape() == "square")
        player.personal_supply_circles = len(route.posts)
        occupy_route(player, route, ["circle"] * len(route.posts))

        self.assertEqual(
            game.legal_action_mask()[route_office_index(game, route, city)].item(),
            0,
        )

    def test_square_only_route_cannot_claim_circle_office(self):
        game = create_headless_game(2, 3, seed=124)
        player = game.current_player
        city = next(
            city
            for city in game.selected_map.cities
            if city.get_next_open_office_shape() == "circle"
            and any(
                all(post.required_shape in (None, "square") for post in route.posts)
                for route in city.routes
            )
        )
        route = next(
            route
            for route in city.routes
            if all(post.required_shape in (None, "square") for post in route.posts)
        )
        occupy_route(player, route, ["square"] * len(route.posts))

        self.assertEqual(
            game.legal_action_mask()[route_office_index(game, route, city)].item(),
            0,
        )

    def test_gold_coin_is_awarded_on_establishment_but_not_office_swap(self):
        game = create_headless_game(1, 3, seed=124)
        player, opponent = game.players[:2]
        city = next(
            city
            for city in game.selected_map.cities
            if len(city.offices) >= 2 and city.offices[0].awards_points
        )
        city.offices[0].controller = player
        city.offices[1].controller = opponent
        score_before = player.score

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(city.swap_offices(player))

        self.assertEqual(player.score, score_before)
        self.assertIs(city.offices[1].controller, player)

    def test_completing_city_updates_completed_city_track_during_route_action(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        city = next(city for city in game.selected_map.cities if len(city.offices) == 1)
        route = city.routes[0]
        occupy_route(player, route)

        self.apply(game, route_office_index(game, route, city))

        self.assertTrue(city.city_is_full())
        self.assertEqual(game.current_full_cities_count, 1)

    def test_tenth_completed_city_ends_after_office_route_action(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        one_office_cities = [city for city in game.selected_map.cities if len(city.offices) == 1]
        target_city = one_office_cities[0]
        other_cities = [city for city in game.selected_map.cities if city is not target_city][:9]
        for city in other_cities:
            for office in city.offices:
                office.controller = game.players[1]
        route = target_city.routes[0]
        occupy_route(player, route)

        self.apply(game, route_office_index(game, route, target_city))

        self.assertTrue(game.game_end)
        self.assertGreaterEqual(game.current_full_cities_count, 10)
        self.assertTrue(all(post.owner is None for post in route.posts))

    def test_route_upgrade_releases_piece_and_returns_entire_route(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        city = next(city for city in game.selected_map.cities if city.name == "Lubeck")
        route = city.routes[0]
        occupy_route(player, route)
        stock_before = player.general_stock_squares
        supply_before = player.personal_supply_squares

        self.apply(game, route_upgrade_index(game, route, city))

        self.assertEqual(player.bank, 4)
        self.assertEqual(player.personal_supply_squares, supply_before + 1)
        self.assertEqual(player.general_stock_squares, stock_before + len(route.posts))
        self.assertTrue(all(post.owner is None for post in route.posts))

    def test_maxed_ability_is_not_a_legal_route_alternative(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        city = next(city for city in game.selected_map.cities if city.name == "Lubeck")
        route = city.routes[0]
        occupy_route(player, route)
        player.bank = 50
        self.assertEqual(
            game.legal_action_mask()[route_upgrade_index(game, route, city)].item(),
            0,
        )

    def test_special_prestige_requires_route_merchant_and_keeps_it_off_stock(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Coellen", "Warburg"}
        )
        city = next(city for city in route.cities if city.name == "Coellen")
        action_index = route_upgrade_index(game, route, city)
        occupy_route(player, route)
        self.assertEqual(game.legal_action_mask()[action_index].item(), 0)

        route.posts[0].reset_post()
        route.posts[0].claim(player, "circle")
        player.personal_supply_squares += 1
        player.personal_supply_circles -= 1
        stock_squares_before = player.general_stock_squares
        stock_circles_before = player.general_stock_circles

        self.apply(game, action_index)

        self.assertEqual(
            game.selected_map.specialprestigepoints.get_special_prestige_points_for_player(player),
            7,
        )
        self.assertEqual(
            player.general_stock_squares,
            stock_squares_before + len(route.posts) - 1,
        )
        self.assertEqual(player.general_stock_circles, stock_circles_before)

    def test_special_prestige_player_may_choose_any_eligible_vacant_value(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        player.privilege = "BLACK"
        route = next(
            route
            for route in game.selected_map.routes
            if {city.name for city in route.cities} == {"Coellen", "Warburg"}
        )
        occupy_route(
            player,
            route,
            ["circle"] + ["square"] * (len(route.posts) - 1),
        )
        base_action = 242 + MAX_ROUTES * 3 + game.selected_map.routes.index(route) * 4
        mask = game.legal_action_mask()
        self.assertEqual(mask[base_action : base_action + 4].tolist(), [1, 1, 1, 1])

        self.apply(game, base_action + 1)

        self.assertEqual(
            game.selected_map.specialprestigepoints.get_special_prestige_points_for_player(player),
            8,
        )

    def test_route_action_applies_exactly_one_step_three_alternative(self):
        game = create_headless_game(1, 3, seed=124)
        player = game.current_player
        city = next(city for city in game.selected_map.cities if city.name == "Lubeck")
        route = city.routes[0]
        occupy_route(player, route)
        points_action = route_points_index(game, route)
        office_action = route_office_index(game, route, city)
        upgrade_action = route_upgrade_index(game, route, city)
        mask = game.legal_action_mask()
        self.assertEqual(mask[points_action].item(), 1)
        self.assertEqual(mask[office_action].item(), 1)
        self.assertEqual(mask[upgrade_action].item(), 1)

        self.apply(game, upgrade_action)

        updated_mask = game.legal_action_mask()
        self.assertEqual(updated_mask[points_action].item(), 0)
        self.assertEqual(updated_mask[office_action].item(), 0)
        self.assertEqual(updated_mask[upgrade_action].item(), 0)

    def test_city_control_tie_is_won_by_rightmost_office(self):
        game = create_headless_game(2, 3, seed=124)
        first, second = game.players[:2]
        city = next(city for city in game.selected_map.cities if len(city.offices) >= 2)
        city.offices[0].controller = first
        city.offices[1].controller = second
        self.assertIs(city.get_controller(), second)

    def test_twentieth_point_ends_only_after_route_action_is_complete(self):
        game = create_headless_game(2, 3, seed=124)
        actor, opponent = game.players[:2]
        route = game.selected_map.routes[0]
        occupy_route(actor, route)
        route.cities[0].offices[0].controller = opponent
        opponent.score = 19

        self.apply(game, route_points_index(game, route))

        self.assertEqual(opponent.score, 20)
        self.assertTrue(game.game_end)
        self.assertTrue(all(post.owner is None for post in route.posts))

    def test_east_west_connection_requires_active_players_continuous_offices(self):
        for map_num in range(1, 4):
            game = create_headless_game(map_num, 3, seed=124)
            player = game.current_player
            start_name, end_name = game.selected_map.east_west_cities
            path = self.city_path(game, start_name, end_name)
            for city in path:
                city.offices[0].controller = player

            self.assertTrue(game.has_east_west_connection(start_name, end_name))
            if len(path) > 2:
                path[len(path) // 2].offices[0].controller = game.players[1]
                self.assertFalse(game.has_east_west_connection(start_name, end_name))

    def test_first_three_east_west_completions_score_seven_four_two_once(self):
        game = create_headless_game(1, 3, seed=124)
        start_name, end_name = game.selected_map.east_west_cities
        path = self.city_path(game, start_name, end_name)

        for player, expected_points in zip(game.players, (7, 4, 2)):
            for city in path:
                open_office = next(
                    (office for office in city.offices if office.controller is None),
                    None,
                )
                if open_office is None:
                    city.create_new_office("WHITE").controller = player
                else:
                    open_office.controller = player
            game.current_player = player
            game.current_player_index = game.players.index(player)
            before = player.score
            with contextlib.redirect_stdout(io.StringIO()):
                game.check_for_east_west_connection()
                game.check_for_east_west_connection()
            self.assertEqual(player.score, before + expected_points)

        self.assertEqual(game.east_west_completed_count, 3)
        self.assertEqual(game.players_who_completed_east_west, set(game.players))


if __name__ == "__main__":
    unittest.main()
