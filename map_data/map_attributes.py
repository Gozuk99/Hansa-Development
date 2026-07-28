# map_attributes.py
import random
from map_data.constants import BLACK, CIRCLE_RADIUS, SQUARE_SIZE, BUFFER, SPACING, TAN, COLOR_NAMES, BLACK, WHITE, ORANGE, PINK, PRIVILEGE_COLORS, DARK_GREEN, DARK_BLUE, BLACKISH_BROWN

class Map:
    STANDARD_BONUS_MARKER_SUPPLY = {
        "PlaceAdjacent": 3,
        "SwapOffice": 2,
        "Move3": 1,
        "UpgradeAbility": 2,
        "3Actions": 2,
        "4Actions": 2,
    }
    PROMO_BONUS_MARKERS = {
        "ExchangeBonusMarker": 2,
        "Tribute4EstablishingTP": 2,
        "BlockTradeRoute": 2,
    }

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else random.Random()
        # This should never change
        self.initial_bonus_types = ['Move3', 'SwapOffice', 'PlaceAdjacent']
        self.permanent_bm_types = ['MoveAny2', '+1Priv', 'ClaimGreenCity', "Place2TradesmenFromRoute", "Place2ScotlandOrWales"]
        self.bonus_marker_pool = []
        self.place_new_bonus_marker = False
        self.specialprestigepoints = None
        # Prepare the starting bonus markers
        self.assign_bm_pool_default()

    def assign_starting_bonus_markers(self):
         # Ensure we shuffle the initial bonus types to randomize the assignment
        self.rng.shuffle(self.initial_bonus_types)

        for route in self.routes:
            # Check if the route has a bonus marker
            if route.has_bonus_marker:
                if self.initial_bonus_types:  # Check if there are still bonus types available
                    bm_type = self.initial_bonus_types.pop()
                    # print(f"Assigning bonus marker: {bm_type} to route between {route.cities[0].name} and {route.cities[1].name}")
                    route.assign_map_new_bonus_marker(bm_type)
                else:
                    print(f"Ran out of initial bonus types to assign for route between {route.cities[0].name} and {route.cities[1].name}")

    def assign_bm_pool_default(self):
        for bm_type, count in self.STANDARD_BONUS_MARKER_SUPPLY.items():
            self.bonus_marker_pool.extend([bm_type] * count)
        self.rng.shuffle(self.bonus_marker_pool)

    def configure_bonus_marker_supply(self, marker_types):
        """Install an explicit twelve-marker supply, including optional promos."""
        marker_types = list(marker_types)
        if len(marker_types) != 12:
            raise ValueError("Bonus-marker supply must contain exactly 12 markers")

        allowed_counts = {
            **self.STANDARD_BONUS_MARKER_SUPPLY,
            **self.PROMO_BONUS_MARKERS,
        }
        for marker_type in set(marker_types):
            if marker_type not in allowed_counts:
                raise ValueError(f"Unknown bonus-marker type: {marker_type}")
            count = marker_types.count(marker_type)
            if count > allowed_counts[marker_type]:
                raise ValueError(
                    f"Too many {marker_type} markers: {count} > {allowed_counts[marker_type]}"
                )

        self.bonus_marker_pool = marker_types
        self.rng.shuffle(self.bonus_marker_pool)
    
class City:
    def __init__(self, name, x_pos, y_pos, color):
        self.name = name
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.color = color
        self.routes = []
        self.controller = None  # Player controlling the city
        self.offices = []  # List of offices within the city
        self.width = 0
        self.height = 0
        self.midpoint = (0, 0)  # Initialize midpoint with (0, 0)
        self.upgrade_city_type = []
        self.tributed_players = [None, None, None, None]

    def add_route(self, route):
        self.routes.append(route)

    def assign_upgrade_type(self, upgrade_type):
        if upgrade_type not in self.upgrade_city_type:
            self.upgrade_city_type.append(upgrade_type)

    def add_office(self, office):
        self.offices.append(office)

    def update_next_open_office_ownership(self, game):
        player = game.current_player
        color = player.color

        for office in self.offices:
            if office.controller is None:
                office.controller = player
                office.color = color
                if office.awards_points:
                    player.score += office.awards_points
                break

        # Update privilege count only if the player is gaining control for the first time
        if self.name == "Cardiff" and game.cardiff_priv != player:
            game.cardiff_priv = player
            player.brown_priv_count += 1
        elif self.name == "Carlisle" and game.carlisle_priv != player:
            game.carlisle_priv = player
            player.blue_priv_count += 1
        elif self.name == "London" and game.london_priv != player:
            game.london_priv = player
            player.london_priv_count += 1

    def update_city_size_based_on_offices(self):
        num_offices = len(self.offices)
        num_circle_offices = sum(1 for office in self.offices if office.shape == "circle")
        num_square_offices = num_offices - num_circle_offices
        
        # Calculate the rectangle dimensions based on the actual number of offices
        rect_width = (
            num_circle_offices * (CIRCLE_RADIUS * 2) +  # Total width of circle offices
            num_square_offices * SQUARE_SIZE +  # Total width of square offices
            2 * BUFFER +  # Buffer at the beginning and end
            SPACING*(num_offices - 1)
        )
        rect_height = max(CIRCLE_RADIUS * 2, SQUARE_SIZE) + BUFFER * 2
        
        self.width = rect_width
        self.height = rect_height
        self.midpoint = (self.x_pos + rect_width / 2, self.y_pos + rect_height / 2)

    def get_controller(self):
        if not self.offices:
            print(f"ERROR: No offices in {self.name}, therefore no controller.")
            return None  # No offices in the city
        
        # Count the number of offices controlled by each player
        player_counts = {}
        for office in self.offices:
            if office.controller:
                player_counts[office.controller] = player_counts.get(office.controller, 0) + 1

        if not player_counts:
            # print(f"No players control any offices in {self.name}.")
            return None  # No offices controlled by any player in the city

        # Determine the player with the maximum number of offices controlled
        max_controlled_offices = max(player_counts.values())
        players_with_max_offices = [player for player, count in player_counts.items() if count == max_controlled_offices]

        # If one player has more offices than the others, they control the city
        if len(players_with_max_offices) == 1:
            self.controller = players_with_max_offices[0]
            # print(f"Player {COLOR_NAMES[self.controller.color]} controls the most offices in {self.name}.")
            return self.controller

        # If there's a tie for the number of offices, find the rightmost player among those tied
        rightmost_office_index = -1
        for player in players_with_max_offices:
            for i, office in reversed(list(enumerate(self.offices))):
                if office.controller == player and i > rightmost_office_index:
                    rightmost_office_index = i
                    self.controller = player

        # print(f"There is a tie. Rightmost player among tied players in {self.name} is Player {COLOR_NAMES[self.controller.color]}.")
        return self.controller
    
    def has_empty_office(self):
        for office in self.offices:
            if office.controller == None:
                return True
        return False

    def has_required_piece_shape(self, player, route):
        """Returns True if the player has the required piece shape on the route to claim an office in the city."""
        required_shape = self.get_next_open_office_shape()

        return any(post.owner_piece_shape == required_shape and post.owner == player for post in route.posts)
    
    def get_next_open_office_shape(self):
        """Return the shape of the next open office in the city. If all offices are claimed, return None."""
        for office in self.offices:
            if office.is_open():
                return office.shape
        return None
    
    def get_next_open_office_color(self):
        """Return the color of the next open office in the city. If all offices are claimed, return None."""
        for office in self.offices:
            if office.is_open():
                return office.color
        return None
    
    def eligible_swap_pairs(self, current_player):
        pairs = []
        for left_index in range(len(self.offices) - 1):
            left = self.offices[left_index]
            right = self.offices[left_index + 1]
            if left.place_adjacent_office or right.place_adjacent_office:
                continue
            if left.controller is None or right.controller is None:
                continue
            if left.controller is right.controller:
                continue
            if current_player in (left.controller, right.controller):
                pairs.append((left_index, left_index + 1))
        return pairs

    def check_if_eligible_to_swap_offices(self, current_player):
        return bool(self.eligible_swap_pairs(current_player))

    def swap_office_pair(self, current_player, pair):
        if pair not in self.eligible_swap_pairs(current_player):
            return False
        left_index, right_index = pair
        self.offices[left_index].controller, self.offices[right_index].controller = (
            self.offices[right_index].controller,
            self.offices[left_index].controller,
        )
        return True

    def swap_offices(self, current_player):
        pairs = self.eligible_swap_pairs(current_player)
        return bool(pairs and self.swap_office_pair(current_player, pairs[0]))

    def can_claim_additional_office(self, player, route, shape):
        standard_offices = [
            office for office in self.offices if not office.place_adjacent_office
        ]
        return (
            bool(standard_offices)
            and standard_offices[0].controller is not None
            and any(
                post.owner is player and post.owner_piece_shape == shape
                for post in route.posts
            )
        )

    def claim_office_with_bonus_marker(self, player, shape="square"):
        # Check if the player has the 'PlaceAdjacent' bonus marker and can't claim the next office
        # Use the bonus marker to create a new office to the left of the leftmost office
        new_office = self.create_new_office(player.color, shape)
        new_office.controller = player
        new_office.color = player.color
        new_office.place_adjacent_office = True

        if self.color == DARK_GREEN:
            # Check the number of owned offices and remove the last one if there are less than 6
            num_owned_offices = sum(1 for office in self.offices if office.controller is not None)
            if num_owned_offices < 6:
                self.offices.pop()

        self.update_city_size_based_on_offices()

        # Update player's bonus markers by removing the first 'PlaceAdjacent'
        place_adjacent_bm = next((bm for bm in player.bonus_markers if bm.type == 'PlaceAdjacent'), None)
        if place_adjacent_bm:
            # Move the used bonus marker to the used list
            player.used_bonus_markers.append(place_adjacent_bm)
            player.bonus_markers.remove(place_adjacent_bm)

        # Notify that the bonus marker was used to place adjacent
        print(f"{COLOR_NAMES[player.color]} used 'PlaceAdjacent' bonus marker to claim a new office in {self.name}.")
        
    def city_is_full(self):
        if self.color == DARK_GREEN:
            # For DARK_GREEN cities, the city is full if any office is occupied
            return any(office.controller is not None for office in self.offices)
        else:
            # For other cities, all offices must be occupied to be considered full
            return self.city_all_offices_occupied()
        
    def city_all_offices_occupied(self):
        return all(office.controller is not None for office in self.offices)
    
    def create_new_office(self, color, shape="square"):
        # Create a new office to the left of the leftmost office
        new_office = Office(shape, color, awards_points=False)
        self.offices.insert(0, new_office)  # Insert the new office at the beginning of the list
        return new_office
        
    def has_office_controlled_by(self, player):
        return any(office.controller == player for office in self.offices)
    
    def has_office_owned_by(self, player):
        for office in self.offices:
            if office.controller == player:
                return True
        return False
    
    def claim_green_city(self, game):
        if game.current_player.personal_supply_squares == 0 and game.current_player.personal_supply_circles == 0:
            print(f"Cannot claim GREEN City: {self.name}, because you have no Tradesmen in your Personal Supply")
            return False
        else:
            if self.city_all_offices_occupied():
                #create a new office
                #append it to city.offices
                self.add_office(Office("square", "WHITE", 0))

            self.update_next_open_office_ownership(game)
            
            # Remove a square if available, otherwise remove a circle
            if game.current_player.personal_supply_squares > 0:
                game.current_player.personal_supply_squares -= 1
            elif game.current_player.personal_supply_circles > 0:
                game.current_player.personal_supply_circles -= 1
            
            print(f"Claimed office in GREEN City: {self.name}")
            return True

class Upgrade:
    def __init__(self, city_name, upgrade_type, x_pos, y_pos, width, height):
        self.city_name = city_name
        self.upgrade_type = upgrade_type
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height

        self.circle_data = [
            {"color": WHITE, "value": 7, "owner": None},
            {"color": ORANGE, "value": 8, "owner": None},
            {"color": PINK, "value": 9, "owner": None},
            {"color": BLACK, "value": 11, "owner": None}
        ]

    def get_special_prestige_points_for_player(self, player):
        return sum(circle['value'] for circle in self.circle_data if circle['owner'] == player)

    def can_claim_prestige(self, player, value=None):
        player_privilege_index = PRIVILEGE_COLORS.index(player.privilege)
        for circle in self.circle_data:
            circle_color_name = COLOR_NAMES.get(circle["color"])
            if (
                (value is None or circle["value"] == value)
                and
                circle["owner"] is None
                and circle_color_name in PRIVILEGE_COLORS
                and player_privilege_index >= PRIVILEGE_COLORS.index(circle_color_name)
            ):
                return True
        return False

    def claim_prestige(self, player, value):
        if not self.can_claim_prestige(player, value):
            return False
        circle = next(
            circle for circle in self.circle_data if circle["value"] == value
        )
        circle["owner"] = player
        circle["color"] = player.color
        return True
    
    def claim_highest_prestige(self, player):
        # Log player details and privileges
        print(f"Player's Color: {COLOR_NAMES[player.color]}, Player's Privilege: {player.privilege}")

        # Player's privilege color
        player_privilege_color = player.privilege  # This should directly return one of the values in PRIVILEGE_COLORS
            
        # Check if the player's privilege allows claiming the circle
        try:
            player_privilege_index = PRIVILEGE_COLORS.index(player_privilege_color)
        except ValueError:
            print(f"Player's privilege color {player_privilege_color} not found in PRIVILEGE_COLORS.")
            return False

        # Check each circle in decreasing order of value
        for circle in sorted(self.circle_data, key=lambda x: x["value"], reverse=True):
            # Log circle details
            print(f"Checking Circle: {circle}")

            # Map RGB to its string name
            circle_color_name = COLOR_NAMES.get(circle["color"])
            if circle_color_name not in PRIVILEGE_COLORS:
                # print(f"Circle color {circle_color_name} not in PRIVILEGE_COLORS. Skipping...")
                continue  # Skip circles with colors not in PRIVILEGE_COLORS

            # Circle's privilege index
            circle_privilege_index = PRIVILEGE_COLORS.index(circle_color_name)
                
            if circle["owner"] is None and player_privilege_index >= circle_privilege_index:
                circle["owner"] = player
                # Change the circle color to the player's RGB to indicate ownership
                circle["color"] = player.color
                print(f"Circle claimed by player. Circle color changed to {COLOR_NAMES[player.color]}.")
                return True
        return False
        
class Office:
    def __init__(self, shape, color, awards_points=0):
        self.shape = shape  # "circle" or "square"
        self.color = color
        self.awards_points = awards_points
        self.controller = None  # Initialize controller as None
        self.place_adjacent_office = False
    def is_open(self):
        """Return True if the office is unclaimed."""
        return self.controller is None

class Route:
    def __init__(self, cities, num_posts, has_bonus_marker=False, permanent_bm_type=None, required_circles=0, color=WHITE, region=None):
        self.cities = cities
        for city in cities:
            city.add_route(self)
        self.num_posts = num_posts
        self.has_bonus_marker = has_bonus_marker
        self.has_permanent_bm_type = permanent_bm_type
        self.bonus_marker = None
        self.permanent_bonus_marker = None
        self.required_circles = required_circles  # Number of posts that must be circles
        self.color = color
        self.region = region
        self.posts = self.create_posts()
        self.tribute_owners = []
        self.block_marker_owners = []

        if self.has_permanent_bm_type:
            self.assign_map_permanent_bonus_marker(self.has_permanent_bm_type)

    def create_posts(self, buffer=0.12):
        city1, city2 = self.cities
        posts = []
        # Pre-calculate differences for position calculation
        x_diff = city2.midpoint[0] - city1.midpoint[0]
        y_diff = city2.midpoint[1] - city1.midpoint[1]

        for i in range(1, self.num_posts + 1):
            t = buffer + (1 - 2 * buffer) * (i / (self.num_posts + 1))

            # Calculate position based on interpolated values
            pos = (city1.midpoint[0] + t * x_diff, city1.midpoint[1] + t * y_diff)

            # Determine if the post requires a specific shape
            required_shape = "circle" if i <= self.required_circles else None
            posts.append(Post(pos, required_shape=required_shape, region=self.region))

        return posts

    def find_empty_post(self):
        for post in self.posts:
            if post.owner == None:
                return post
        return None
    
    def has_tradesmen(self):
        for post in self.posts:
            if post.owner is not None:
                # print(f"Route check: Tradesman found at post {self.posts.index(post)+1}/{self.num_posts} on the route between {self.cities[0].name} and {self.cities[1].name}.")
                return True
        # print(f"Route check: No tradesmen found on the route between {self.cities[0].name} and {self.cities[1].name}.")
        return False

    def has_empty_office_in_cities(self):
        city1, city2 = self.cities
        if city1.has_empty_office() or city2.has_empty_office():
            # print(f"Route check: At least one empty office found in cities {city1.name} or {city2.name}.")
            return True
        # print(f"Route check: No empty offices found in cities {city1.name} and {city2.name}.")
        return False
    
    def is_controlled_by(self, player):
        return all(post.owner == player for post in self.posts)

    def is_complete(self):
        for post in self.posts:
            if post.color == BLACK:
                return False
        return True
    
    def assign_map_new_bonus_marker(self, bm_type):
        if not self.bonus_marker:  # Only assign if there's no bonus marker already
            # print(f"Route between {self.cities[0].name} and {self.cities[1].name} is being assigned a bonus marker of type {bm_type}")
            self.bonus_marker = BonusMarker(bm_type)
            self.has_bonus_marker = True
        else:
            print(f"Route between {self.cities[0].name} and {self.cities[1].name} already has a bonus marker assigned")

    def assign_map_permanent_bonus_marker(self, bm_type):
        if not self.bonus_marker:  # Only assign if there's no bonus marker already
            # print(f"Route between {self.cities[0].name} and {self.cities[1].name} is being assigned a bonus marker of type {bm_type}")
            self.permanent_bonus_marker = BonusMarker(bm_type)
        else:
            print(f"Route between {self.cities[0].name} and {self.cities[1].name} already has a bonus marker assigned")

    def contains_a_circle(self):
        for post in self.posts:
            if post.owner_piece_shape == "circle":
                return True
        return False
        
    def establish_tribute_on_route(self, player):
        self.tribute_owners.append(player)
    
    def award_tributes(self, game):
        eligible = [
            player
            for player in self.tribute_owners
            if player.general_stock_squares or player.general_stock_circles
        ]
        game.begin_tribute_income_responses(eligible)
    
    def establish_blocked_route(self, player):
        self.block_marker_owners.append(player)
        for post in self.posts:
            post.blocked_bm = True

class BonusMarker:
    def __init__(self, type, owner=None):
        self.type = type
        self.owner = owner
        self.position = (0, 0)
    
    def is_clicked(self, mouse_pos):
        # Check if the mouse click is within the circle of the bonus marker
        distance_squared = (self.position[0] - mouse_pos[0]) ** 2 + (self.position[1] - mouse_pos[1]) ** 2
        return distance_squared <= CIRCLE_RADIUS ** 2
        
    def handle_swap_office(self, city, player):
        if city.check_if_eligible_to_swap_offices(player):
            print ("Valid City to swap offices")
            city.swap_offices(player)
            return True
        else:
            print ("Invalid City to Swap offices, please try another city.")
            return False
        
    def handle_move3(self, game):
        game.waiting_for_bm_move3 = True
        game.current_player.pieces_to_pickup = 3  # Set the pieces to move to 3 as per the bonus marker
        print("You can now move up to 3 opponent's pieces. Click on an opponent's piece to move it.")
        
    def handle_upgrade_ability(self, upgrade, player):
        if player.perform_upgrade(upgrade.upgrade_type):
            print("Successfully used Upgrade BM")
            return True
        else:
            print("Invalid click when Upgrading via BM")
            return False

    def handle_3_actions(self, current_player):
        current_player.grant_actions(3)

    def handle_4_actions(self, current_player):
        current_player.grant_actions(4)

    def handle_tribute4_establishing_tp(self, route, current_player):
        if current_player.personal_supply_squares <= 0:
            print("Player has no Tradesmen in their personal supply to establish a trade post.")
            return False
        current_player.personal_supply_squares -= 1
        route.establish_tribute_on_route(current_player)
        return True
    
    def handle_block_trade_route(self, route, current_player):
        if current_player.personal_supply_squares <= 0:
            print("Player has no Tradesmen in their personal supply to establish a trade post.")
            return False
        current_player.personal_supply_squares -= 1
        route.establish_blocked_route(current_player)
        return True
    
    def handle_exchange_bonus_marker(self, game):
        used_bm_owner = self.owner
        current_player = game.current_player

        current_player.bonus_markers.append(self)
        used_bm_owner.used_bonus_markers.remove(self)
        self.owner = current_player

        for used_bms in current_player.used_bonus_markers:
            if used_bms.type == "ExchangeBonusMarker":
                used_bm_owner.used_bonus_markers.append(used_bms)
                current_player.used_bonus_markers.remove(used_bms)
                used_bms.owner = used_bm_owner
                break

class Post:
    def __init__(self, position, owner=None, required_shape=None, region=None):
        self.pos = position
        self.owner = owner  # This represents the player who owns the post.
        self.owner_piece_shape = None  # This represents the player who owns the post.
        self.circle_color = TAN
        self.square_color = TAN
        self.required_shape = required_shape  # can be "circle", "square", or None if no specific requirement
        self.region = region
        self.blocked_bm = False

    def reset_post(self):
        self.circle_color = TAN
        self.square_color = TAN
        self.owner = None
        self.owner_piece_shape = None
    
    def valid_post_to_displace_to(self):
        self.circle_color = WHITE
        self.square_color = WHITE

    def is_owned(self):
        return self.owner is not None

    def can_be_claimed_by(self, shape):
        return self.owner is None and (self.required_shape is None or self.required_shape == shape)
    
    def is_valid_for_displacement(self, player):
        if self.owner is not None:
            return False  # The post is already claimed
        if self.required_shape is None or player.has_general_stock(self.required_shape):
            return True  # The post is empty and the player has the required shape
        return False

    def claim(self, player, shape):
        if shape == "circle":
            self.circle_color = player.color
            self.square_color = TAN
        else:
            self.square_color = player.color
            self.circle_color = TAN
        self.owner = player
        self.owner_piece_shape = shape
    
    def DEBUG_print_post_details(self):
        print(f"Post Details!!!")
        print(f"Required Shape: {self.required_shape}")
        print(f"Post Position: {self.pos}")
        print(f"Owner Piece Shape: {self.owner_piece_shape}")
        
        # Check if owner is None before trying to access its color
        owner_color = 'None' if self.owner is None else COLOR_NAMES[self.owner.color]
        print(f"Owner: {owner_color}")

        # Check if circle_color is None before trying to access its color name
        circle_color_name = 'None' if self.circle_color is None else COLOR_NAMES.get(self.circle_color, 'Unknown')
        print(f"Circle Color: {circle_color_name}")

        # Check if square_color is None before trying to access its color name
        square_color_name = 'None' if self.square_color is None else COLOR_NAMES.get(self.square_color, 'Unknown')
        print(f"Square Color: {square_color_name}")
