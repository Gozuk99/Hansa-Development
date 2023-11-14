# player_attributes.py

import sys
from map_data.constants import CITY_KEYS_MAX_VALUES, ACTIONS_MAX_VALUES, PRIVILEGE_COLORS, BOOK_OF_KNOWLEDGE_MAX_VALUES, BANK_MAX_VALUES, COLOR_NAMES, UPGRADE_METHODS_MAP, UPGRADE_MAX_VALUES

class Player:
    def __init__(self, color, order):
        self.color = color
        self.score = 0  # Initial score
        self.final_score = 0
        # The silver plate to store bonus markers for the end of the turn
        self.bonus_markers = []
        self.used_bonus_markers = []
        self.holding_pieces = []
        self.pieces_to_place = None

        # The available abilities with their starting values
        self.keys = 1
        self.privilege = "WHITE"
        self.book = 2
        self.actions_index = 0
        self.actions = ACTIONS_MAX_VALUES[5]
        self.actions_remaining = ACTIONS_MAX_VALUES[0]
        self.bank = 3

        # Setting the order-based attributes for general_stock and personal_supply
        self.general_stock_squares = 7 - order  
        self.general_stock_circles = 0  

        # Personal supply attributes (distinct circles and squares)
        self.personal_supply_squares = 4 + order # Squares in personal supply
        self.personal_supply_circles = 1 # Each player always starts with 1 circle in personal supply

        self.board = None

    def add_bonus_marker(self, marker):
        self.bonus_markers.append(marker)
    
    def start_move(self):
        # Start a new move by setting pieces to place equal to book value
        self.pieces_to_place = self.book
        self.holding_pieces = []
        print(f"Starting move: you can move up to {self.book} pieces.")

    def pick_up_piece(self, post):
        # Pick up a piece from the post if under the limit
        if self.pieces_to_place > 0 and post.owner_piece_shape:
            self.holding_pieces.append((post.owner_piece_shape, post.owner))
            self.pieces_to_place -= 1
            print(f"Picked up Player {COLOR_NAMES[post.owner.color]}'s {post.owner_piece_shape}. {self.pieces_to_place} moves left.")  # Assuming owner has a 'color' attribute
            post.reset_post()
        else:
            message = "No more pieces can be picked up this turn." if self.pieces_to_place <= 0 else "This post is empty."
            print(message)

    def place_piece(self, post, button):
        if not self.holding_pieces:
            print("No pieces to place.")
            return

        # Retrieve the shape and the owner of the next piece to be placed
        shape_to_place, owner_to_place = self.holding_pieces[0]

        # Determine the shape based on the button clicked
        shape_clicked = 'circle' if button == 3 else 'square'

        if shape_to_place == shape_clicked:
            # The shape matches, so place it and remove from holding_pieces
            print(f"Please place Player {COLOR_NAMES[owner_to_place.color]}'s {shape_to_place}.")  # Assuming owner has a 'color' attribute
            post.claim(owner_to_place, shape_to_place)  # Note: The owner should be passed to the claim method
            self.holding_pieces.pop(0)
            print(f"Placed Player {COLOR_NAMES[owner_to_place.color]}'s {shape_to_place} on the board.")
        else:
            print(f"The next piece to place must be Player {COLOR_NAMES[owner_to_place.color]}'s {shape_to_place}.")

    def finish_move(self):
        # End the move process if no pieces are being held
        if not self.holding_pieces:
            print("Move completed.")
            self.actions_remaining -= 1  # Deduct an action for the move
            self.pieces_to_place = None  # Clear the move commitment
        else:
            print("You still have pieces to place. Finish your move.")

    def upgrade_keys(self):
        print("Upgrade Keys called")
        if self.keys < max(CITY_KEYS_MAX_VALUES):
            self.keys = CITY_KEYS_MAX_VALUES[CITY_KEYS_MAX_VALUES.index(self.keys) + 1]
        else:
            print("City_Keys is already at its maximum level!")

    def upgrade_actions(self):
        # Check if we can upgrade
        if self.actions_index + 1 < len(ACTIONS_MAX_VALUES):
            previous_actions = self.actions  # store the previous value of actions

            self.actions_index += 1
            self.actions = ACTIONS_MAX_VALUES[self.actions_index]
            
            # If the new value of actions is greater than the previous one, increment actions_remaining by 1
            if self.actions > previous_actions:
                self.actions_remaining += 1
        else:
            print("Actions are already at maximum!")

    def upgrade_privilege(self):
        print("Upgrade priv called")
        if PRIVILEGE_COLORS.index(self.privilege) < len(PRIVILEGE_COLORS) - 1:
            self.privilege = PRIVILEGE_COLORS[PRIVILEGE_COLORS.index(self.privilege) + 1]
        else:
            print("Privilege is already at its maximum level!")

    def upgrade_book(self):
        print("Upgrade book called")
        if self.book < max(BOOK_OF_KNOWLEDGE_MAX_VALUES):
            self.book = BOOK_OF_KNOWLEDGE_MAX_VALUES[BOOK_OF_KNOWLEDGE_MAX_VALUES.index(self.book) + 1]
        else:
            print("Book_of_Knowledge is already at its maximum level!")

    def upgrade_bank(self):
        print("Upgrade bank called")
        current_index = BANK_MAX_VALUES.index(self.bank)
        if current_index < len(BANK_MAX_VALUES) - 1:
            self.bank = BANK_MAX_VALUES[current_index + 1]
        else:
            print("Bank is already at its maximum level!")

    def perform_upgrade(self, upgrade_type):
        method_name = UPGRADE_METHODS_MAP[upgrade_type.lower()]
        current_value = getattr(self, upgrade_type.lower())
        max_value = UPGRADE_MAX_VALUES.get(upgrade_type.lower())

        if current_value != max_value:
            upgrade_function = getattr(self, method_name)
            upgrade_function()

            if upgrade_type.lower() in ["keys", "privilege", "actions", "bank"]:
                self.personal_supply_squares += 1
            elif upgrade_type.lower() == "book":
                self.personal_supply_circles += 1

            # If this is called from a bonus marker, you might not want to adjust actions or switch player
            return True
        else:
            print(f"{upgrade_type} is already at its maximum value for player {COLOR_NAMES[self.color]}.")
            return False

    def has_unlocked_key(self, index):
        return self.keys > index

    def has_unlocked_privilege(self, index):
        privileges_order = ["WHITE", "ORANGE", "PINK", "BLACK"]
        return privileges_order.index(self.privilege) >= index

    def has_unlocked_book(self, index):
        return self.book > index + 1

    def has_unlocked_action(self, index):
        return self.actions_index > index

    def has_unlocked_bank(self, index):
        # If the player's bank is "C", then all slots are unlocked
        if self.bank == "C":
            return True
        # For other bank values, compare only if the value at the given index is an integer
        if isinstance(BANK_MAX_VALUES[index], int):
            return self.bank >= BANK_MAX_VALUES[index]
        return False  # For any other cases
    
    def income_action(self, num_squares=0, num_circles=0):
        if num_circles <= self.general_stock_circles and num_squares <= self.general_stock_squares:
            self.general_stock_circles -= num_circles
            self.personal_supply_circles += num_circles
            
            self.general_stock_squares -= num_squares
            self.personal_supply_squares += num_squares
        elif self.general_stock_circles + self.general_stock_squares > self.bank:
            raise Exception("The sum of general stock circles and squares exceeds the bank value.")
    
    def income_action_based_on_circle_count(self, max_circles, bank, general_stock_squares):
        button_labels = []
        
        if bank == "C":
            label = f"{general_stock_squares}S/{max_circles}C"
            button_labels.append(label)
            return button_labels

        for i in range(max_circles + 1):
            circles = i
            squares = min(bank - circles, general_stock_squares)
            if 0 <= squares:
                label = f"{squares}S/{circles}C"
                button_labels.append(label)
        return button_labels

    def player_can_claim_office(self, office_color):
        """Check if a player can claim an office of the specified color."""
        allowed_office_colors = PRIVILEGE_COLORS[:PRIVILEGE_COLORS.index(self.privilege) + 1]
        return office_color in allowed_office_colors

class DisplacedPlayer:
    def __init__(self):
        self.reset_displaced_player()

    def reset_displaced_player(self):
        self.player = None
        self.displaced_shape = None
        self.played_displaced_shape = False
        self.total_pieces_to_place = 0

    def populate_displaced_player(self, player, displaced_shape):
        self.player = player
        self.displaced_shape = displaced_shape
        if self.displaced_shape == "square":
            self.total_pieces_to_place = 2
        elif self.displaced_shape == "circle":
            self.total_pieces_to_place = 3
        else:
            sys.exit()
            
    def all_pieces_placed(self):
        return self.total_pieces_to_place == 0 and self.played_displaced_shape

class PlayerBoard:
    def __init__(self, x, y, player):
        self.x = x
        self.y = y
        self.player = player
        self.width = 790  # adjust based on your requirement
        self.height = 200  # adjust based on your requirement
        self.start_x = 0
        self.actions_y = 0
        self.circle_buttons = []
        self.button_labels = []

    def income_action_based_on_circle_count(self, idx):
        label = self.button_labels[idx]
        num_squares, num_circles = [int(x[:-1]) for x in label.split('/')]
        self.player.income_action(num_squares, num_circles)