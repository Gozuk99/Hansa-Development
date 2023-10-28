# player_attributes.py

import pygame
import sys
from map_data.constants import SQUARE_SIZE, CIRCLE_RADIUS, WHITE, ORANGE, PINK, BLACK, CITY_KEYS_MAX_VALUES, ACTIONS_MAX_VALUES, PRIVILEGE_COLORS, BOOK_OF_KNOWLEDGE_MAX_VALUES, BANK_MAX_VALUES
from drawing.drawing_utils import draw_shape, draw_text

class Player:
    def __init__(self, color, order):
        self.color = color
        self.score = 0  # Initial score
        # The silver plate to store bonus markers for the end of the turn
        self.bonus_markers = []

        # The available abilities with their starting values
        self.keys = 1
        self.privilege = "white"
        self.book = 2
        self.actions = 2
        self.bank = 3

        # Setting the order-based attributes for general_stock and personal_supply
        self.general_stock_squares = 7 - order  
        self.general_stock_circles = 0  

        # Personal supply attributes (distinct circles and squares)
        self.personal_supply_squares = 4 + order # Squares in personal supply
        self.personal_supply_circles = 1 # Each player always starts with 1 circle in personal supply

    def add_bonus_marker(self, marker):
        self.bonus_markers.append(marker)

    def upgrade_keys(self):
        if self.keys < max(CITY_KEYS_MAX_VALUES):
            self.keys = CITY_KEYS_MAX_VALUES[CITY_KEYS_MAX_VALUES.index(self.keys) + 1]
        else:
            print("City_Keys is already at its maximum level!")

    def upgrade_actions(self):
        if self.actions < max(ACTIONS_MAX_VALUES):
            self.actions = ACTIONS_MAX_VALUES[ACTIONS_MAX_VALUES.index(self.actions) + 1]
        else:
            print("Actions are already at their maximum level!")

    def upgrade_privilege(self):
        if PRIVILEGE_COLORS.index(self.privilege) < len(PRIVILEGE_COLORS) - 1:
            self.privilege = PRIVILEGE_COLORS[PRIVILEGE_COLORS.index(self.privilege) + 1]
        else:
            print("Privilege is already at its maximum level!")

    def upgrade_book(self):
        if self.book < max(BOOK_OF_KNOWLEDGE_MAX_VALUES):
            self.book = BOOK_OF_KNOWLEDGE_MAX_VALUES[BOOK_OF_KNOWLEDGE_MAX_VALUES.index(self.book) + 1]
        else:
            print("Book_of_Knowledge is already at its maximum level!")

    def upgrade_bank(self):
        current_index = BANK_MAX_VALUES.index(self.bank)
        if current_index < len(BANK_MAX_VALUES) - 1:
            self.bank = BANK_MAX_VALUES[current_index + 1]
        else:
            print("Bank is already at its maximum level!")

    def income_action(self, num_squares=0, num_circles=0):
        if num_circles <= self.general_stock_circles and num_squares <= self.general_stock_squares:
            self.general_stock_circles -= num_circles
            self.personal_supply_circles += num_circles
            
            self.general_stock_squares -= num_squares
            self.personal_supply_squares += num_squares
        elif self.general_stock_circles + self.general_stock_squares > self.bank:
            raise Exception("The sum of general stock circles and squares exceeds the bank value.")
            
class PlayerBoard:
    def __init__(self, x, y, player_color, player):
        self.x = x
        self.y = y
        self.player_color = player_color  # Set player color
        self.player = player
        self.width = 790  # adjust based on your requirement
        self.height = 200  # adjust based on your requirement
        self.font = pygame.font.SysFont(None, 32)

    def income_action_based_on_circle_count(self, idx):
        # Retrieve the button label based on the index
        label = self.button_labels[idx]
        num_squares, num_circles = [int(x[:-1]) for x in label.split('/')]
        
        # Update the player's state
        self.player.income_action(num_squares, num_circles)

    def draw_circle_selection_buttons(self, window):
        # Starting position for the Income label
        income_x = self.x + self.width - 220
        income_y = self.y + self.height - 70

        # Button dimensions and spacing
        button_width = 60
        button_height = 30
        horizontal_spacing = 70  # Spacing between columns
        vertical_spacing = 35    # Spacing between rows

        # Clear any previously drawn buttons
        self.circle_buttons = []
        
        # Use the `draw_text` function for centering on both X and Y axis
        font = pygame.font.SysFont(None, 28)

        # Calculate the center coordinates of the button
        button_center_x = income_x + button_width / 2
        button_center_y = income_y + button_height / 2

        # Draw the Income text centered on both axes
        draw_text(window, "Income:", button_center_x, button_center_y, font, BLACK, centered=True)

        # Logic for determining which buttons to display based on the conditions
        max_circles = min(self.player.general_stock_circles, 4)  # Considering up to 4 circles
        self.button_labels = []

        for i in range(max_circles + 1):
            circles = i
            squares = min(self.player.bank - circles, self.player.general_stock_squares)

            if 0 <= squares:
                label = f"{squares}S/{circles}C"
                self.button_labels.append(label)

        for i, label in enumerate(self.button_labels):
            if i == 0:
                button_x = income_x
                button_y = income_y + vertical_spacing
            elif i == 1:
                button_x = income_x + horizontal_spacing
                button_y = income_y
            elif i == 2:
                button_x = income_x + horizontal_spacing
                button_y = income_y + vertical_spacing
            elif i == 3:
                button_x = income_x + 2 * horizontal_spacing
                button_y = income_y
            elif i == 4:
                button_x = income_x + 2 * horizontal_spacing
                button_y = income_y + vertical_spacing

            # Draw the button
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            self.circle_buttons.append(button_rect)
            draw_shape(window, 'rectangle', BLACK, button_x, button_y, button_width, button_height)

            # Label for the button
            draw_text(window, label, button_x + (button_width // 2), button_y + (button_height // 2), pygame.font.SysFont(None, 24), WHITE, centered=True)

    def draw_general_stock(self, window):
        x_offset = self.x + 650  # Adjust this value based on exact positioning
        y_offset = self.y + 10  # Adjust this value based on exact positioning

        # Draw 'GS:' text
        font = pygame.font.SysFont(None, 36)
        draw_text(window, 'GS:', x_offset, y_offset, font, BLACK)
        
        # Draw square for squares count with the number inside
        x_offset += 50
        draw_shape(window, 'rectangle', self.player.color, x_offset, y_offset, 40, 40)
        draw_text(window, str(self.player.general_stock_squares), x_offset + 20, y_offset + 20, font, (0, 0, 0), centered=True)  # Assuming black color for numbers

        # Draw circle next to square for circles count with number inside
        x_offset += 60
        draw_shape(window, 'circle', self.player.color, x_offset, y_offset + 20, 20) # Assuming radius is half of square side
        draw_text(window, str(self.player.general_stock_circles), x_offset, y_offset + 20, font, (0, 0, 0), centered=True)  # Assuming black color for numbers
        
    def draw_personal_supply(self, window):
        x_offset = self.x + 650  # Adjust this value based on exact positioning
        y_offset = self.y + 80  # Positioning it further down than GS for clarity

        # Draw 'PS:' text
        font = pygame.font.SysFont(None, 36)
        draw_text(window, 'PS:', x_offset, y_offset, font, BLACK)

        # Draw square for squares count in personal supply with the number inside
        x_offset += 50
        draw_shape(window, 'rectangle', self.player.color, x_offset, y_offset, 40, 40)
        draw_text(window, str(self.player.personal_supply_squares), x_offset + 20, y_offset + 20, font, (0, 0, 0), centered=True)

        # Draw circle next to square for circles count in personal supply with number inside
        x_offset += 60
        draw_shape(window, 'circle', self.player.color, x_offset, y_offset + 20, 20)
        draw_text(window, str(self.player.personal_supply_circles), x_offset, y_offset + 20, font, (0, 0, 0), centered=True)

    def draw(self, window, current_player):
        # Draw board background with player color
        draw_shape(window, "rectangle", self.player_color, self.x, self.y, self.width, self.height)

        # Draw "City Keys" section as diamonds
        for i, value in enumerate(CITY_KEYS_MAX_VALUES):
            points = [
                (self.x + 10 + i * (SQUARE_SIZE + 5), self.y + 10 + SQUARE_SIZE // 2),
                (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, self.y + 10),
                (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE, self.y + 10 + SQUARE_SIZE // 2),
                (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, self.y + 10 + SQUARE_SIZE)
            ]
            draw_shape(window, "polygon", WHITE, None, None, points=points)
            draw_text(window, str(value), self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, self.y + 10 + SQUARE_SIZE // 2, self.font, BLACK, centered=True)
        
        draw_text(window, "Keys", self.x + 10, self.y + 10 + SQUARE_SIZE + 5, self.font, BLACK)

        # Adjusted "Privilegium" section with buffer moved further down
        privilege_y = self.y + 10 + 2*SQUARE_SIZE + 10  # adjusting spacing
        colors = [WHITE, ORANGE, PINK, BLACK]
        for i, color in enumerate(colors):
            draw_shape(window, "rectangle", color, self.x + 10 + i*(SQUARE_SIZE + 5), privilege_y, width=SQUARE_SIZE, height=SQUARE_SIZE)
        
        draw_text(window, "Privilege", self.x + 10, privilege_y + SQUARE_SIZE + 5, self.font, BLACK)
            
        # Draw "Liber Sophiae" section (circles)
        start_x = self.x + 10 + len(colors)*50 + 5*(len(colors)-1) + 10  # buffer after Privilege

        circle_label = self.font.render("Liber Sophiae", True, BLACK)
        circle_section_width = len(BOOK_OF_KNOWLEDGE_MAX_VALUES) * (CIRCLE_RADIUS*2 + 5)  # total width of all circles combined
        circle_label_width = circle_label.get_width()

        # Adjust start_x to center the "Liber Sophiae" section
        start_x += (circle_section_width - circle_label_width) // 2

        for i, value in enumerate(BOOK_OF_KNOWLEDGE_MAX_VALUES):
            draw_shape(window, "circle", WHITE, start_x + i*(CIRCLE_RADIUS*2 + 5), self.y + 10 + CIRCLE_RADIUS, width=CIRCLE_RADIUS)
            draw_text(window, str(value), start_x + i*(CIRCLE_RADIUS*2 + 5), self.y + 10 + CIRCLE_RADIUS, self.font, BLACK, centered=True)

        draw_text(window, "Liber Sophiae", start_x-CIRCLE_RADIUS, self.y + 10 + CIRCLE_RADIUS*2 + 5, self.font, BLACK)
            
        start_x += len(BOOK_OF_KNOWLEDGE_MAX_VALUES) * (CIRCLE_RADIUS * 2 + 5) + 10
        # Calculate y-position for the "Actiones" section first
        actions_y = self.y + 10

        # Draw "Actiones" section (squares)
        for i, value in enumerate(ACTIONS_MAX_VALUES):
            draw_shape(window, "rectangle", WHITE, start_x + i*(SQUARE_SIZE + 5), actions_y, width=SQUARE_SIZE, height=SQUARE_SIZE)
            draw_text(window, str(value), start_x + i*(SQUARE_SIZE + 5) + SQUARE_SIZE // 2, actions_y + SQUARE_SIZE // 2, self.font, BLACK, centered=True)

        draw_text(window, "Actiones", start_x, actions_y + SQUARE_SIZE + 5, self.font, BLACK)

        # Calculate y-position for the "Bank" section based on "Actiones" section height
        bank_y = actions_y + SQUARE_SIZE + 5 + self.font.get_height() + 5

        # Draw "Bank" section (squares)
        for i, value in enumerate(BANK_MAX_VALUES):
            draw_shape(window, "rectangle", WHITE, start_x + i*(SQUARE_SIZE + 5), bank_y, width=SQUARE_SIZE, height=SQUARE_SIZE)
            draw_text(window, str(value), start_x + i*(SQUARE_SIZE + 5) + SQUARE_SIZE // 2, bank_y + SQUARE_SIZE // 2, self.font, BLACK, centered=True)

        draw_text(window, "Bank", start_x, bank_y + SQUARE_SIZE + 5, self.font, BLACK)

        self.draw_general_stock(window)
        self.draw_personal_supply(window)

        # Inside the draw() method
        if self.player == current_player:
            self.draw_circle_selection_buttons(window)

    # More methods for PlayerBoard operations can be added here.