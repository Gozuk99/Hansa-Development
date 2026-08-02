import random

from game.action_codec import ActionCodecError, DEFAULT_ACTION_CODEC
from game.action_schema import TILE_TYPES
from game.action_execution import execute_action
from game.game_actions import InvalidActionError
from game.legal_actions import get_legal_actions
from game.setup import validate_game_configuration
from game.turn_state import TurnPhase, TurnStateError
from map_data.map1 import Map1
from map_data.map2 import Map2
from map_data.map3 import Map3
from map_data.constants import COLOR_NAMES, WHITE, GREEN, BLUE, PURPLE, RED, YELLOW
from player_info.player_attributes import Player, DisplacedPlayer, PlayerBoard, UPGRADE_MAX_VALUES


class Game:
    def __init__(
        self,
        map_num,
        num_players,
        seed=None,
        interactive_errors=True,
        use_mission_cards=False,
        use_emperors_favour=False,
        bonus_marker_supply=None,
    ):
        validate_game_configuration(map_num, num_players)
        if use_mission_cards and map_num != 1:
            raise ValueError("Mission Cards can only be enabled on map 1")

        self.seed = seed
        self.rng = random.Random(seed)
        self.ai_model = None
        self.interactive_errors = interactive_errors
        self.map_num = map_num
        self.use_mission_cards = use_mission_cards
        self.use_emperors_favour = use_emperors_favour
        self.selected_map = self.assign_map(map_num, num_players)
        if bonus_marker_supply is not None:
            self.selected_map.configure_bonus_marker_supply(bonus_marker_supply)
        self.num_players = num_players

        self.players = self.create_players(num_players)
        self.current_player_index = 0
        self.current_player = self.players[self.current_player_index]
        self.active_player = self.current_player_index
        self.turn_number = 1
        self.round_number = 1
        self.replace_bonus_marker = 0
        self.pending_bonus_markers = []
        self.bonus_pool_exhausted_during_claim = False

        self.displaced_player = DisplacedPlayer()
        self.waiting_for_displaced_player = False

        self.east_west_completed_count = 0
        self.players_who_completed_east_west = (
            set()
        )  # Track players who have completed the connection

        self.waiting_for_bm_swap_office = False
        self.waiting_for_bm_place_adjacent = False
        self.waiting_for_bm_upgrade_ability = False
        self.waiting_for_bm_move_any_2 = False
        self.waiting_for_bm_move3 = False

        self.waiting_for_bm_exchange_bm = False
        self.pending_exchange_marker = None
        self.exchange_target_player = None
        self.waiting_for_bm_tribute_trading_post = False
        self.waiting_for_bm_block_trade_route = False

        self.waiting_for_bm_green_city = False
        self.waiting_for_place2_from_route = False
        self.pending_route_piece_choices = []
        self.waiting_for_place2_in_scotland_or_wales = False
        self.pending_britannia_place2 = False

        self.tile_to_buy = None
        self.waiting_for_buy_tile_with_bm = False
        self.first_bm_to_spend_on_tile = None
        self.pending_income_favour_owner = None
        self.pending_tribute_income_owners = []

        self.original_route_of_displacement = None
        self.all_empty_posts = []
        self.tile_pool = []
        if self.use_emperors_favour:
            self.initialize_tile_pool()
        self.tile_rects = []

        self.current_full_cities_count = 0

        self.cardiff_priv = None
        self.carlisle_priv = None
        self.london_priv = None

        self.DisplaceAnywhereOwner = None
        self.OneActionOwner = None
        self.OneIncomeIfOthersIncomeOwner = None
        self.OneDisplacedPieceOwner = None
        self.FourPtsPerOwnedCityOwner = None
        self.SevenPtsPerCompletedAbilityOwner = None

        self.game_end = False
        self.game_end_pending_immediate_resolution = False

    def create_players(self, num_players):
        colors = [GREEN, BLUE, PURPLE, RED, YELLOW]
        players = []

        for i, color in enumerate(colors[:num_players]):
            new_player = Player(color, i + 1)
            new_player.board = PlayerBoard(
                self.selected_map.map_width, i * 220, new_player
            )  # Create and assign the board directly here
            new_player.start_turn()

            if self.use_mission_cards:
                self.selected_map.assign_mission_cards(
                    new_player
                )  # Assign a mission card to the player

            players.append(new_player)

        return players

    def initialize_tile_pool(self):
        tiles = list(TILE_TYPES)
        self.rng.shuffle(tiles)
        self.tile_pool.extend(tiles[: self.num_players])

    def assign_map(self, map_num, num_players):
        # Logic to assign a map based on map_num
        if map_num == 1:
            return Map1(num_players, rng=self.rng)
        elif map_num == 2:
            return Map2(rng=self.rng)
        elif map_num == 3:
            return Map3(num_players, rng=self.rng)

    @property
    def pending_workflows(self):
        workflows = []
        if self.waiting_for_displaced_player:
            workflows.append(TurnPhase.DISPLACEMENT)
        if self.waiting_for_buy_tile_with_bm:
            workflows.append(TurnPhase.BUY_TILE_PAYMENT)
        if self.pending_income_favour_owner is not None:
            workflows.append(TurnPhase.INCOME_FAVOUR_RESPONSE)
        if self.pending_tribute_income_owners:
            workflows.append(TurnPhase.TRIBUTE_INCOME_RESPONSE)
        if self.waiting_for_bm_place_adjacent:
            workflows.append(TurnPhase.PLACE_ADJACENT_ROUTE)
        if self.pending_route_piece_choices:
            workflows.append(TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION)
        if self.pending_britannia_place2:
            workflows.append(TurnPhase.PERMANENT_ROUTE_PIECE_SELECTION)
        if self.replace_bonus_marker > 0 and self.current_player.actions_remaining == 0:
            workflows.append(TurnPhase.REPLACE_BONUS_MARKERS)

        bonus_pending = any(
            (
                self.waiting_for_bm_swap_office,
                self.waiting_for_bm_upgrade_ability,
                self.waiting_for_bm_move_any_2,
                self.waiting_for_bm_move3,
                self.waiting_for_bm_exchange_bm,
                self.waiting_for_bm_tribute_trading_post,
                self.waiting_for_bm_block_trade_route,
                self.waiting_for_bm_green_city,
                self.waiting_for_place2_from_route,
                self.waiting_for_place2_in_scotland_or_wales,
            )
        )
        if bonus_pending:
            workflows.append(TurnPhase.BONUS_MARKER_CHOICE)
        elif self.current_player.holding_pieces:
            workflows.append(TurnPhase.MOVE_PIECES)

        return tuple(workflows)

    @property
    def turn_phase(self):
        if self.game_end:
            return TurnPhase.GAME_OVER
        workflows = self.pending_workflows
        immediate_workflows = tuple(
            workflow for workflow in workflows if workflow != TurnPhase.REPLACE_BONUS_MARKERS
        )
        if len(immediate_workflows) > 1:
            names = ", ".join(workflow.value for workflow in immediate_workflows)
            raise TurnStateError(f"Conflicting pending workflows: {names}")
        if immediate_workflows:
            return immediate_workflows[0]
        if TurnPhase.REPLACE_BONUS_MARKERS in workflows:
            return TurnPhase.REPLACE_BONUS_MARKERS
        if self.current_player.actions_remaining == 0:
            return TurnPhase.TURN_COMPLETE
        return TurnPhase.ACTIONS

    def advance_turn(self):
        if self.turn_phase != TurnPhase.TURN_COMPLETE:
            raise TurnStateError(f"Cannot advance player during phase {self.turn_phase.value}")

        previous_player = self.current_player
        previous_player.ending_turn = False
        previous_player.actions_granted_this_turn = 0
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.turn_number += 1
        if self.current_player_index == 0:
            self.round_number += 1
        self.current_player = self.players[self.current_player_index]
        extra_actions = 1 if self.OneActionOwner == self.current_player else 0
        self.current_player.start_turn(extra_actions=extra_actions)
        self.active_player = self.current_player_index

        if self.map_num == 3:
            self.current_player.refresh_map3_priv_actions(self)

    def switch_player_if_needed(self):
        if self.turn_phase == TurnPhase.TURN_COMPLETE:
            self.advance_turn()
            return True
        return False

    def get_legal_actions(self):
        """Return authoritative structured legal interactions."""
        return get_legal_actions(self)

    def ai_action_mask(self):
        """Return the authoritative 768-entry AI action mask."""
        return DEFAULT_ACTION_CODEC.create_mask(self.get_legal_actions())

    def apply_structured_action(self, action):
        """Validate and execute one structured interaction."""
        if action not in self.get_legal_actions():
            raise InvalidActionError(f"Structured action is not legal: {action!r}")
        execute_action(self, action)

    def apply_ai_action(self, action_index):
        """Decode and execute one action from the 768-entry AI schema."""
        try:
            action = DEFAULT_ACTION_CODEC.decode(action_index)
        except ActionCodecError as error:
            raise InvalidActionError(str(error)) from error
        self.apply_structured_action(action)

    def apply_action(self, action_index):
        """Apply one codec-backed action for GUI and manual callers."""
        self.apply_ai_action(action_index)

    def reset_valid_posts(self):
        for post in self.all_empty_posts:
            post.reset_post()
        self.all_empty_posts.clear()

    def check_brown_blue_priv(self, route):
        if route.region is not None:
            # Check for Wales region
            if route.region == "Wales":
                if not (
                    self.current_player.brown_priv_count > 0
                    or self.current_player.london_priv_count > 0
                ):
                    return False

            # Check for Scotland region
            elif route.region == "Scotland":
                if not (
                    self.current_player.blue_priv_count > 0
                    or self.current_player.london_priv_count > 0
                ):
                    return False
        return True

    def consume_region_privilege(self, route):
        if route.region == "Wales":
            if self.current_player.brown_priv_count > 0:
                self.current_player.brown_priv_count -= 1
            elif self.current_player.london_priv_count > 0:
                self.current_player.london_priv_count -= 1
            else:
                raise TurnStateError("No Wales placement permission remains")
        elif route.region == "Scotland":
            if self.current_player.blue_priv_count > 0:
                self.current_player.blue_priv_count -= 1
            elif self.current_player.london_priv_count > 0:
                self.current_player.london_priv_count -= 1
            else:
                raise TurnStateError("No Scotland placement permission remains")

    def begin_income_favour_response(self, income_player):
        """Offer the optional income benefit after another player's Income action."""
        owner = self.OneIncomeIfOthersIncomeOwner
        if (
            owner is not None
            and owner is not income_player
            and (owner.general_stock_squares or owner.general_stock_circles)
        ):
            self.pending_income_favour_owner = owner
            self.active_player = owner.order - 1

    def resolve_income_favour(self, shape=None):
        owner = self.pending_income_favour_owner
        if owner is None:
            raise TurnStateError("No Emperor's Favour income response is pending")
        if shape is not None:
            owner.add_1_income(shape)
        self.pending_income_favour_owner = None
        self.active_player = self.current_player_index

    def begin_tribute_income_responses(self, owners):
        self.pending_tribute_income_owners.extend(owners)
        if self.pending_tribute_income_owners:
            self.active_player = self.pending_tribute_income_owners[0].order - 1

    def resolve_tribute_income(self, num_circles):
        if not self.pending_tribute_income_owners:
            raise TurnStateError("No tribute income is pending")
        owner = self.pending_tribute_income_owners[0]
        total_available = owner.general_stock_squares + owner.general_stock_circles
        amount = min(2, total_available)
        num_squares = amount - num_circles
        if (
            num_circles < 0
            or num_circles > owner.general_stock_circles
            or num_squares < 0
            or num_squares > owner.general_stock_squares
        ):
            raise TurnStateError("Selected tribute-income composition is unavailable")
        owner.income_action(num_squares, num_circles, tribute_income=True)
        self.pending_tribute_income_owners.pop(0)
        self.active_player = (
            self.pending_tribute_income_owners[0].order - 1
            if self.pending_tribute_income_owners
            else self.current_player_index
        )
        self.complete_deferred_game_end_if_ready()

    def complete_deferred_game_end_if_ready(self):
        if not self.game_end_pending_immediate_resolution:
            return
        immediate = [
            workflow
            for workflow in self.pending_workflows
            if workflow != TurnPhase.REPLACE_BONUS_MARKERS
        ]
        if not immediate:
            self.game_end_pending_immediate_resolution = False
            self.check_for_game_end()

    def check_for_east_west_connection(self):
        if self.current_player in self.players_who_completed_east_west:
            return

        if not self.check_if_player_has_matching_offices_in_east_west(
            self.selected_map.east_west_cities[0], self.selected_map.east_west_cities[1]
        ):
            return

        if self.has_east_west_connection(
            self.selected_map.east_west_cities[0], self.selected_map.east_west_cities[1]
        ):
            east_west_points = [7, 4, 2]
            if self.east_west_completed_count < len(east_west_points):
                awarded_points = east_west_points[self.east_west_completed_count]
                self.current_player.score += awarded_points
                self.east_west_completed_count += 1
                self.players_who_completed_east_west.add(self.current_player)

    def check_if_player_has_matching_offices_in_east_west(self, start_city_name, end_city_name):
        start_city = next(
            (city for city in self.selected_map.cities if city.name == start_city_name), None
        )
        end_city = next(
            (city for city in self.selected_map.cities if city.name == end_city_name), None
        )

        if not start_city or not end_city:
            return False

        return start_city.has_office_controlled_by(
            self.current_player
        ) and end_city.has_office_controlled_by(self.current_player)

    def has_east_west_connection(self, start_city_name, end_city_name, visited=None):
        # This is a recursive depth-first search (DFS) algorithm.
        if visited is None:
            visited = set()

        start_city = next(
            (city for city in self.selected_map.cities if city.name == start_city_name), None
        )
        end_city = next(
            (city for city in self.selected_map.cities if city.name == end_city_name), None
        )

        # Check if both cities exist in the game.
        if start_city is None or end_city is None:
            return False
        if not start_city.has_office_controlled_by(self.current_player):
            return False
        if not end_city.has_office_controlled_by(self.current_player):
            return False

        # If we've reached the end city, return True
        if start_city == end_city:
            return True

        # Mark the start city as visited
        visited.add(start_city)

        # Go through each route connected to the start city
        for route in start_city.routes:
            # Check all cities connected to this route
            for connected_city in route.cities:
                # Skip if we've already visited this city or if the connected city doesn't have the player's office
                if connected_city in visited or not connected_city.has_office_controlled_by(
                    self.current_player
                ):
                    continue

                # Recursively check if the connected city leads to the end city
                if self.has_east_west_connection(connected_city.name, end_city_name, visited):
                    return True

        # If none of the routes lead to the end city, return False
        return False

    def get_bonus_marker_points(self, total_bms):
        if total_bms == 1:
            return 1
        elif 2 <= total_bms <= 3:
            return 3
        elif 4 <= total_bms <= 5:
            return 6
        elif 6 <= total_bms <= 7:
            return 10
        elif 8 <= total_bms <= 9:
            return 15
        elif total_bms >= 10:
            return 21
        else:
            return 0

        # 1 initial points
        # 2 fully developed abilities
        # 3 prestige points for total bonus markers collected
        #  1-1, 2or3-3, 4or5-6, 6or7-10, 8or9-15, 10+ - 21
        # 4 specialprestigepoints 7/8/9/11
        # 5 prestige points for cities, 2 per control
        # 6 largest network x key

    def dfs_network_size(self, player, city, visited_cities):
        if city in visited_cities:
            return 0  # This city is already part of the current network

        visited_cities.add(city)
        offices_in_city = sum(1 for office in city.offices if office.controller == player)
        network_size = offices_in_city  # Count the number of offices instead of just the city

        for route in city.routes:
            for connected_city in route.cities:
                if (
                    connected_city != city
                    and connected_city.has_office_owned_by(player)
                    and connected_city not in visited_cities
                ):
                    network_size += self.dfs_network_size(player, connected_city, visited_cities)

        return network_size

    def calculate_largest_network(self, player):
        largest_network = 0
        all_visited_cities = set()

        for city in self.selected_map.cities:
            if city.has_office_owned_by(player) and city not in all_visited_cities:
                visited_cities = set()
                network_size = self.dfs_network_size(player, city, visited_cities)
                largest_network = max(largest_network, network_size)
                all_visited_cities.update(visited_cities)

        return largest_network

    def projected_score_breakdown(self, player, britannia_region_points=None):
        """Calculate one player's authoritative score if the game ended now."""
        if britannia_region_points is None:
            britannia_region_points = (
                self.calculate_britannia_region_points() if self.map_num == 3 else {}
            )

        ability_points = 0
        for ability in ["privilege", "book", "actions", "bank"]:
            if getattr(player, ability) == UPGRADE_MAX_VALUES[ability]:
                ability_points += 4
                if self.SevenPtsPerCompletedAbilityOwner == player:
                    ability_points += 3

        total_bms = len(player.bonus_markers) + len(player.used_bonus_markers)
        special_prestige_points = 0
        if self.selected_map.specialprestigepoints is not None:
            special_prestige_points = (
                self.selected_map.specialprestigepoints.get_special_prestige_points_for_player(
                    player
                )
            )

        city_control_points = 0
        for city in self.selected_map.cities:
            if city.determine_controller() == player:
                city_control_points += 2
                if self.FourPtsPerOwnedCityOwner == player:
                    city_control_points += 2

        breakdown = {
            "Initial Points": player.score,
            "Ability Points": ability_points,
            "Bonus Marker Points": self.get_bonus_marker_points(total_bms),
            "Special Prestige Points": special_prestige_points,
            "City Control Points": city_control_points,
            "Largest Network Points": self.calculate_largest_network(player) * player.keys,
            "Britannia Region Points": britannia_region_points.get(player, 0),
        }
        if self.use_mission_cards and player.mission_card:
            breakdown["Mission City Points"] = self.get_mission_card_points(player)
        return breakdown

    def projected_scores(self):
        """Return final-score projections in seat order without ending the game."""
        britannia = self.calculate_britannia_region_points() if self.map_num == 3 else {}
        return tuple(
            sum(self.projected_score_breakdown(player, britannia).values())
            for player in self.players
        )

    def finalize_end_of_game_points(self):
        britannia = self.calculate_britannia_region_points() if self.map_num == 3 else {}
        for player in self.players:
            breakdown = self.projected_score_breakdown(player, britannia)
            player.final_score = sum(breakdown.values())
            player.final_score_breakdown = breakdown
            if player.final_score < player.score:
                raise TurnStateError(
                    f"Final score for {COLOR_NAMES[player.color]} is below the in-game score"
                )

    def calculate_britannia_region_points(self):
        """Award the Britannia 7/4/2 ladders for Wales and, on 4–5p, Scotland."""
        totals = {player: 0 for player in self.players}
        regions = ["Wales"]
        if self.num_players > 3:
            regions.append("Scotland")

        city_names_by_region = {
            region: {
                city.name
                for route in self.selected_map.routes
                if route.region == region
                for city in route.cities
            }
            | {"IsleOfMan"}
            for region in regions
        }
        awards = (7, 4, 2)
        for region in regions:
            standings = []
            for player in self.players:
                cities = [
                    city
                    for city in self.selected_map.cities
                    if city.name in city_names_by_region[region]
                ]
                controlled = sum(city.determine_controller() == player for city in cities)
                offices = sum(
                    office.controller == player for city in cities for office in city.offices
                )
                if offices:
                    standings.append((player, controlled, offices))
            standings.sort(key=lambda item: (item[1], item[2]), reverse=True)

            position = 0
            index = 0
            while index < len(standings) and position < len(awards):
                metric = standings[index][1:]
                tied = []
                while index < len(standings) and standings[index][1:] == metric:
                    tied.append(standings[index][0])
                    index += 1
                available = awards[position : min(position + len(tied), len(awards))]
                shared = sum(available) // len(tied)
                for player in tied:
                    totals[player] += shared
                position += len(tied)
        return totals

    def get_mission_card_points(self, player):
        """Score one point per listed city occupied, plus five for controlling all three."""
        if not self.use_mission_cards or not player.mission_card:
            return 0

        mission_cities = [
            city for city in self.selected_map.cities if city.name in player.mission_card
        ]
        occupied = sum(city.has_office_owned_by(player) for city in mission_cities)
        controls_all = len(mission_cities) == len(player.mission_card) and all(
            city.determine_controller() == player for city in mission_cities
        )
        return occupied + (5 if controls_all else 0)

    def check_for_game_end(self):
        self.current_full_cities_count = sum(
            1 for city in self.selected_map.cities if city.city_is_full()
        )

        # Check if the bonus marker pool is empty or any player has reached the score threshold
        end_conditions_met = (
            self.bonus_pool_exhausted_during_claim
            or any(player.score >= 20 for player in self.players)
            or self.current_full_cities_count >= self.selected_map.max_full_cities
        )

        if end_conditions_met:
            immediate = [
                workflow
                for workflow in self.pending_workflows
                if workflow != TurnPhase.REPLACE_BONUS_MARKERS
            ]
            if immediate:
                self.game_end_pending_immediate_resolution = True
                return
            self.current_player.forfeit_remaining_actions()
            # Finalize points before determining the winner
            self.finalize_end_of_game_points()
            self.game_end = True

    def end_the_game(self):
        highest_score = max(player.final_score for player in self.players)
        tied = [player for player in self.players if player.final_score == highest_score]
        if len(tied) <= 1:
            return tied

        least_developed_actions = min(player.actions_index for player in tied)
        tied = [player for player in tied if player.actions_index == least_developed_actions]
        if len(tied) <= 1:
            return tied

        largest_network_score = max(
            player.final_score_breakdown.get("Largest Network Points", 0) for player in tied
        )
        return [
            player
            for player in tied
            if player.final_score_breakdown.get("Largest Network Points", 0)
            == largest_network_score
        ]
