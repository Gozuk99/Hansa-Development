# player_attributes.py

import pygame
from map_data.constants import SQUARE_SIZE, CIRCLE_RADIUS, WHITE, ORANGE, PINK, BLACK, CITY_KEYS_MAX_VALUES, ACTIONS_MAX_VALUES, PRIVILEGE_COLORS, BOOK_OF_KNOWLEDGE_MAX_VALUES, BANK_MAX_VALUES

class Player:
    def __init__(self, color):
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

    # Other relevant methods like calculate_score, move_tradesman, etc. can be added here

class PlayerBoard:
    def __init__(self, x, y, player_color):
        self.x = x
        self.y = y
        self.player_color = player_color  # Set player color
        self.width = 800  # adjust based on your requirement
        self.height = 200  # adjust based on your requirement
        self.font = pygame.font.SysFont(None, 36)
    
    def draw(self, window):
        # Draw board background with player color
        pygame.draw.rect(window, self.player_color, (self.x, self.y, self.width, self.height))

        # Draw "City Keys" section as diamonds
        for i, value in enumerate(CITY_KEYS_MAX_VALUES):
            points = [
                (self.x + 10 + i * (SQUARE_SIZE + 5), self.y + 10 + SQUARE_SIZE // 2),
                (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, self.y + 10),
                (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE, self.y + 10 + SQUARE_SIZE // 2),
                (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, self.y + 10 + SQUARE_SIZE)
            ]
            pygame.draw.polygon(window, WHITE, points)
            text = self.font.render(str(value), True, BLACK)
            window.blit(text, (self.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2 - text.get_width() // 2, self.y + 10 + SQUARE_SIZE // 2 - text.get_height() // 2))
        
        keys_label = self.font.render("Keys", True, BLACK)
        window.blit(keys_label, (self.x + 10, self.y + 10 + SQUARE_SIZE + 5))

        # Adjusted "Privilegium" section with buffer moved further down
        privilege_y = self.y + 10 + 2*SQUARE_SIZE + 10  # adjusting spacing
        colors = [WHITE, ORANGE, PINK, BLACK]
        for i, color in enumerate(colors):
            pygame.draw.rect(window, color, (self.x + 10 + i * (SQUARE_SIZE + 5), privilege_y, SQUARE_SIZE, SQUARE_SIZE))
        
        privilege_label = self.font.render("Privilege", True, BLACK)
        window.blit(privilege_label, (self.x + 10, privilege_y + SQUARE_SIZE + 5))  # 5 pixels below the Privilege squares
            
        # Draw "Liber Sophiae" section (circles)
        start_x = self.x + 10 + len(colors)*50 + 5*(len(colors)-1) + 10  # buffer after Privilege

        circle_label = self.font.render("Liber Sophiae", True, BLACK)
        circle_section_width = len(BOOK_OF_KNOWLEDGE_MAX_VALUES) * (CIRCLE_RADIUS*2 + 5)  # total width of all circles combined
        circle_label_width = circle_label.get_width()

        # Adjust start_x to center the "Liber Sophiae" section
        start_x += (circle_section_width - circle_label_width) // 2

        for i, value in enumerate(BOOK_OF_KNOWLEDGE_MAX_VALUES):
            pygame.draw.circle(window, WHITE, (start_x + i*(CIRCLE_RADIUS*2 + 5), self.y + 10 + CIRCLE_RADIUS), CIRCLE_RADIUS)
            text = self.font.render(str(value), True, BLACK)
            text_x = start_x + i*(CIRCLE_RADIUS*2 + 5) - text.get_width() // 2
            text_y = self.y + 10 + CIRCLE_RADIUS - text.get_height() // 2
            window.blit(text, (text_x, text_y))

        window.blit(circle_label, (start_x-CIRCLE_RADIUS, self.y + 10 + CIRCLE_RADIUS*2 + 5))
            
        start_x += len(BOOK_OF_KNOWLEDGE_MAX_VALUES) * (CIRCLE_RADIUS * 2 + 5) + 10
        # Calculate y-position for the "Actiones" section first
        actions_y = self.y + 10

        # Draw "Actiones" section (squares)
        for i, value in enumerate(ACTIONS_MAX_VALUES):
            pygame.draw.rect(window, WHITE, (start_x + i*(SQUARE_SIZE + 5), actions_y, SQUARE_SIZE, SQUARE_SIZE))
            text = self.font.render(str(value), True, BLACK)
            # Center the text inside the square
            text_x = start_x + i*(SQUARE_SIZE + 5) + (SQUARE_SIZE - text.get_width()) // 2
            text_y = actions_y + (SQUARE_SIZE - text.get_height()) // 2
            window.blit(text, (text_x, text_y))

        actions_label = self.font.render("Actiones", True, BLACK)
        window.blit(actions_label, (start_x, actions_y + SQUARE_SIZE + 5))

        # Calculate y-position for the "Bank" section based on "Actiones" section height
        bank_y = actions_y + SQUARE_SIZE + 5 + actions_label.get_height() + 5

        # Draw "Bank" section (squares)
        for i, value in enumerate(BANK_MAX_VALUES):
            pygame.draw.rect(window, WHITE, (start_x + i*(SQUARE_SIZE + 5), bank_y, SQUARE_SIZE, SQUARE_SIZE))
            text = self.font.render(str(value), True, BLACK)
            # Center the text inside the square
            text_x = start_x + i*(SQUARE_SIZE + 5) + (SQUARE_SIZE - text.get_width()) // 2
            text_y = bank_y + (SQUARE_SIZE - text.get_height()) // 2
            window.blit(text, (text_x, text_y))

        bank_label = self.font.render("Bank", True, BLACK)
        window.blit(bank_label, (start_x, bank_y + SQUARE_SIZE + 5))

    # More methods for PlayerBoard operations can be added here.