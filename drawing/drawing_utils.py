import pygame
from map_data.constants import TAN, COLOR_NAMES, WHITE, ORANGE, PINK, BLACK, YELLOW, BUFFER, SQUARE_SIZE, SPACING, CIRCLE_RADIUS, CITY_KEYS_MAX_VALUES, ACTIONS_MAX_VALUES, PRIVILEGE_COLORS, BOOK_OF_KNOWLEDGE_MAX_VALUES, BANK_MAX_VALUES, BLUE

pygame.init()

FONT_LARGE = pygame.font.Font(None, 36)  
FONT_PLAYERBOARD = pygame.font.SysFont(None, 32)
FONT_SMALL = pygame.font.Font(None, 24)  
BORDER_WIDTH = 2 #black outline

def draw_shape(window, shape_to_draw, color, x, y, width=None, height=None, points=None):
    if shape_to_draw == "circle":
        if width is None:
            raise ValueError("For circle, 'width' is required as the radius.")
        pygame.draw.circle(window, color, (x, y), width)
        pygame.draw.circle(window, BLACK, (x, y), width, BORDER_WIDTH)  # Border

    elif shape_to_draw == "rectangle":
        if width is None or height is None:
            raise ValueError("For rectangle, both 'width' and 'height' are required.")
        pygame.draw.rect(window, color, (x, y, width, height))
        pygame.draw.rect(window, BLACK, (x, y, width, height), BORDER_WIDTH)  # Border

    elif shape_to_draw == "polygon":
        if points is None:
            raise ValueError("For polygon, 'points' is required as a list of points.")
        pygame.draw.polygon(window, color, points)
        pygame.draw.polygon(window, BLACK, points, BORDER_WIDTH)  # Border
    else:
        raise ValueError("Invalid shape_to_draw value.")
    
def draw_text(window, text, x, y, font, color=BLACK, centered=False):
    text_surface = font.render(text, True, color)

    if centered:
        text_rect = text_surface.get_rect(center=(x, y))
    else:
        text_rect = text_surface.get_rect(topleft=(x, y))

    window.blit(text_surface, text_rect)

def draw_upgrades(win, selected_map):
    for upgrade_type in selected_map.upgrade_cities:
        draw_upgrade_on_map(win, upgrade_type)
    draw_special_prestige_points(win, selected_map.specialprestigepoints)

def draw_upgrade_on_map(window, upgrade_type):
    draw_shape(window, "rectangle", YELLOW, upgrade_type.x_pos, upgrade_type.y_pos, width=upgrade_type.width, height=upgrade_type.height)

    # If upgrade type is 'SpecialPrestigePoints', handle it differently
    if upgrade_type.upgrade_type == "SpecialPrestigePoints":
        upgrade_type.draw_special_prestige_points(window)
        return

    x_font_centered = upgrade_type.x_pos + (upgrade_type.width / 2)
    y_font_centered = upgrade_type.y_pos + (upgrade_type.height / 2)
    draw_text(window, upgrade_type.upgrade_type, x_font_centered, y_font_centered, FONT_SMALL, color=BLACK, centered=True)

def draw_special_prestige_points(window, upgrade_type):
    draw_shape(window, "rectangle", YELLOW, upgrade_type.x_pos, upgrade_type.y_pos, width=upgrade_type.width, height=upgrade_type.height)

    # Define the total width of all circles and spaces combined
    total_width = (CIRCLE_RADIUS * 2) * 4 + (SPACING * 3)

    # Define starting position for the circles
    start_x = upgrade_type.x_pos + (upgrade_type.width - total_width) / 2 + CIRCLE_RADIUS  # Adjust the starting position
    start_y = upgrade_type.y_pos + upgrade_type.height / 2  # This centers the circle vertically in the rectangle

    for circle in upgrade_type.circle_data:
        # Draw circle with the circle's color (either a privilege color or a player's color)
        pygame.draw.circle(window, circle["color"], (int(start_x), int(start_y)), CIRCLE_RADIUS)

        # Render text
        text_surface = FONT_LARGE.render(str(circle["value"]), True, WHITE if circle["color"] == BLACK else BLACK)
        text_rect = text_surface.get_rect(center=(start_x, start_y))
        window.blit(text_surface, text_rect)

        # Adjust start_x for next circle
        start_x += CIRCLE_RADIUS * 2 + SPACING
    
def draw_completed_cities_indicator(win, game):
    selected_map = game.selected_map
    num_full_cities = game.current_full_cities_count

    # Draw label below the boxes
    draw_text(win, "Completed Cities", selected_map.max_full_cities_x_pos, selected_map.max_full_cities_y_pos + SQUARE_SIZE + 5, FONT_SMALL, BLACK, centered=False)

    # Draw a rectangle for each city and the test value (city index) in each square
    for i in range(selected_map.max_full_cities):
        rect_x = selected_map.max_full_cities_x_pos + (SQUARE_SIZE + SPACING) * i
        rect_y = selected_map.max_full_cities_y_pos
        color = BLACK if i < num_full_cities else WHITE

        # Draw the city rectangle using draw_shape function
        draw_shape(win, "rectangle", color, rect_x, rect_y, SQUARE_SIZE, SQUARE_SIZE)

        # Draw the test value (city index) in each square using draw_text function
        draw_text(win, str(i + 1), rect_x + SQUARE_SIZE // 2, rect_y + SQUARE_SIZE // 2, FONT_SMALL, BLACK, centered=True)

def draw_bonus_markers(win, selected_map):
    routes = selected_map.routes

    win.fill(TAN)

    for route in routes:
        draw_line(win, route.color, route.cities[0].midpoint, route.cities[1].midpoint, 10, 2)
        # Check if the route has a bonus marker and call its draw method
        if route.bonus_marker:
            # Construct the key for the dictionary
            city_pair = tuple(sorted([route.cities[0].name, route.cities[1].name]))
            # Fetch the bonus marker position from the dictionary
            bonus_marker_pos = selected_map.bonus_marker_positions.get(city_pair)

            # If the position exists, call the draw method on the bonus marker
            if bonus_marker_pos:
                draw_board_bonus_markers(win, route.bonus_marker, bonus_marker_pos)
                # print(f"Drew bonus marker between {city_pair[0]} and {city_pair[1]} at position {bonus_marker_pos}")
            # else:
                # print(f"No bonus marker position found for route between {city_pair[0]} and {city_pair[1]}")
        if route.permanent_bonus_marker:
            # Construct the key for the dictionary
            city_pair = tuple(sorted([route.cities[0].name, route.cities[1].name]))
            # Fetch the bonus marker position from the dictionary
            bonus_marker_pos = selected_map.bonus_marker_positions.get(city_pair)

            # If the position exists, call the draw method on the bonus marker
            if bonus_marker_pos:
                draw_board_bonus_markers(win, route.permanent_bonus_marker, bonus_marker_pos, color=BLUE)

def draw_board_bonus_markers(screen, bonus_marker, position, color=BLACK):
    # Draw the bonus marker as a simple shape (e.g., a circle)
    pygame.draw.circle(screen, color, position, 30)
    # Draw the text for the bonus marker type
    font = pygame.font.SysFont(None, 24)
    text = font.render(bonus_marker.type, True, WHITE)  # Render the text with the bonus marker's type
    text_rect = text.get_rect(center=position)  # Get a rect object to center the text inside the circle
    screen.blit(text, text_rect)  # Draw the text to the screen at the specified position

def draw_line(surface, color, start_pos, end_pos, line_width, border_width):
    # Draw the thicker border line first
    pygame.draw.line(surface, BLACK, start_pos, end_pos, line_width + 2 * border_width)
    # Draw the main line on top of the border
    pygame.draw.line(surface, color, start_pos, end_pos, line_width)

def draw_cities_and_offices(win, cities):
    for city in cities:
        draw_city_rectangle(win, city)
        draw_city_offices(win, city)

def draw_city_rectangle(win, city):
    rect_x, rect_y = city.x_pos, city.y_pos
    draw_shape(win, "rectangle", city.color, rect_x, rect_y, city.width, city.height)
    draw_text_below_rectangle(win, city.name, rect_x, rect_y, city.width, city.height)

def draw_text_below_rectangle(win, text, rect_x, rect_y, rect_width, rect_height):
    text_width, text_height = FONT_LARGE.size(text)
    text_x = rect_x + (rect_width - text_width) // 2
    text_y = rect_y + rect_height
    draw_text(win, text, text_x, text_y, FONT_LARGE, BLACK)

def draw_office_awards_points(win, office, center_x, center_y):
    if office.awards_points > 0:
        text = str(office.awards_points)
        text_width, text_height = FONT_LARGE.size(text)
        # Calculate the centered text coordinates
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2
        draw_text(win, text, text_x, text_y, FONT_LARGE, BLACK)

def draw_square_office(win, office, start_x, start_y):
    # Draw the square office
    draw_shape(win, "rectangle", office.color, start_x, start_y, SQUARE_SIZE, SQUARE_SIZE)
    # Calculate the center of the square
    center_x = start_x + SQUARE_SIZE // 2
    center_y = start_y + SQUARE_SIZE // 2
    # Draw the points text centered in the square
    draw_office_awards_points(win, office, center_x, center_y)

def draw_circle_office(win, office, start_x, start_y):
    # Calculate the center of the circle
    center_x = start_x + CIRCLE_RADIUS
    center_y = start_y + CIRCLE_RADIUS
    # Draw the circle office
    draw_shape(win, "circle", office.color, center_x, center_y, CIRCLE_RADIUS)
    # Draw the points text centered in the circle
    draw_office_awards_points(win, office, center_x, center_y)

def draw_city_offices(win, city):
    start_x = city.x_pos + BUFFER
    start_y = city.y_pos + city.height // 2 - SQUARE_SIZE // 2

    for office in city.offices:
        if office.shape == "square":
            draw_square_office(win, office, start_x, start_y)
            start_x += SQUARE_SIZE + SPACING
        else:  # office.shape == "circle"
            draw_circle_office(win, office, start_x, start_y - CIRCLE_RADIUS // 2)  # Adjust Y for circle center
            start_x += CIRCLE_RADIUS * 2 + SPACING

def draw_routes(win, routes):
    for route in routes:
        for post in route.posts:
            draw_route_post(win, post)

def draw_route_post(win, post):
    post_x, post_y = post.pos

    # If the post requires being a circle, draw only a circle
    if post.required_shape == "circle":
        draw_shape(win, "circle", post.circle_color, post_x, post_y, CIRCLE_RADIUS)
    else:
        # Draw a circle and a square (if the post is not owned or owned as a square)
        draw_shape(win, "circle", post.circle_color, post_x, post_y, CIRCLE_RADIUS)
        if post.owner_piece_shape is None or post.owner_piece_shape == "square":
            draw_shape(win, "rectangle", post.square_color, post_x - SQUARE_SIZE // 2, post_y - SQUARE_SIZE // 2, SQUARE_SIZE, SQUARE_SIZE)

def draw_actions_remaining(win, game):
    padding = 5  # Define padding value for spacing around text

    # Calculate text dimensions and positions
    if game.waiting_for_displaced_player:
        text_str = f"{COLOR_NAMES[game.current_player.color]} displaced {COLOR_NAMES[game.displaced_player.player.color]} - waiting for {COLOR_NAMES[game.displaced_player.player.color]} to place {game.displaced_player.total_pieces_to_place} pieces!"
    else:
        text_str = f"Actions: {game.current_player.actions_remaining}"

    combined_text = FONT_LARGE.render(text_str, True, COLOR_NAMES[game.current_player.color])
    text_width = combined_text.get_width() + 2 * padding  # Add padding to width
    text_height = combined_text.get_height() + 2 * padding  # Add padding to height
    text_x = game.selected_map.map_width // 2 - text_width // 2
    text_y = game.selected_map.map_height - 50 - text_height // 2

    # Draw the background for the new text
    draw_shape(win, "rectangle", WHITE, text_x, text_y, text_width, text_height)

    # Draw the new text with padding applied
    win.blit(combined_text, (text_x + padding, text_y + padding))
    pygame.display.update((text_x, text_y, text_width, text_height))

def draw_scoreboard(win, players, start_x, start_y):
    scoreboard_height = len(players) * 20 + 50  # Adjust based on text size and spacing
    scoreboard_width = 200  # Set the width according to your requirements
    scoreboard_rect = pygame.Rect(start_x, start_y, scoreboard_width, scoreboard_height)

    # Fill the scoreboard background
    win.fill(TAN, scoreboard_rect)

    # Draw the scoreboard label at the top
    score_board_label = FONT_LARGE.render("Score Board:", True, BLACK)
    win.blit(score_board_label, scoreboard_rect.topleft)

    # Set initial Y offset for player scores just below the label
    y_offset = score_board_label.get_height() + 5

    # Draw each player's score
    for index, player in enumerate(players):
        score_text = f"Player {index + 1}: {player.score}"
        text_surface = FONT_LARGE.render(score_text, True, player.color)
        text_position = (start_x, start_y + y_offset + (index * 25))  # Y position adjusted here
        win.blit(text_surface, text_position)

def draw_end_turn(win, game):
    start_x = game.selected_map.map_width + 320
    start_y = game.selected_map.map_height - 170
    end_turn_width = 200  # Set the width according to your requirements
    end_turn_height = 70  # Adjust based on text size and spacing

    # Draw the End Turn rectangle background
    draw_shape(win, "rectangle", TAN, start_x, start_y, end_turn_width, end_turn_height)

    # Calculate center position for the End Turn label
    label_center_x = start_x + end_turn_width // 2
    label_center_y = start_y + end_turn_height // 2

    # Draw the End Turn label centered
    draw_text(win, "End Turn", label_center_x, label_center_y, FONT_LARGE, BLACK, centered=True)

def draw_end_game(win, winning_players):
    # Create a semi-transparent surface
    transparent_surface = pygame.Surface(win.get_size(), pygame.SRCALPHA)
    transparent_surface.fill((0, 0, 0, 64))  # 64 here for 25% transparency

    # Blit the semi-transparent surface onto the window
    win.blit(transparent_surface, (0, 0))

    # Now create and blit the end game text
    if len(winning_players) == 1:
        winner_text_str = f"Game Over! {COLOR_NAMES[winning_players[0].color]} wins with {winning_players[0].final_score} points!"
    else:
        winners_str = ", ".join(f"{COLOR_NAMES[player.color]} ({player.final_score} points)" for player in winning_players)
        winner_text_str = f"Game Over! It's a tie: {winners_str}"

    # Create the text surface
    winner_text = FONT_LARGE.render(winner_text_str, True, WHITE)
    text_rect = winner_text.get_rect(center=(win.get_width() // 2, win.get_height() // 2))

    # Blit the text surface onto the window
    win.blit(winner_text, text_rect)

    # Update the display to show the changes
    pygame.display.update()

def draw_get_game_state_button(win, game):
    start_x = game.selected_map.map_width + 320
    start_y = game.selected_map.map_height - 100
    games_state_width = 200  # Set the width according to your requirements
    games_state_height = 70  # Adjust based on text size and spacing

    draw_shape(win, "rectangle", TAN, start_x, start_y, games_state_width, games_state_height)
    # Calculate center position for the End Turn label
    label_center_x = start_x + games_state_width // 2
    label_center_y = start_y + games_state_height // 2

    # Draw the End Turn label centered
    draw_text(win, "Get Game State", label_center_x, label_center_y, FONT_LARGE, BLACK, centered=True)

def draw_bonus_marker_pool(win, game):
    start_x = game.selected_map.map_width + 5
    start_y = game.selected_map.map_height - 170
    bonus_marker_pool_text_box_width = 295  # Set the width according to your requirements
    bonus_marker_pool_text_box_height = 140  # Adjust based on text size and spacing

    draw_shape(win, "rectangle", TAN, start_x, start_y, bonus_marker_pool_text_box_width, bonus_marker_pool_text_box_height)

    # Define label details
    label = f"Bonus Marker Pool - ({len(game.selected_map.bonus_marker_pool)} left):"
    label_x = start_x + bonus_marker_pool_text_box_width // 2
    label_y = start_y + 10  # 10 pixels from the top of the box

    # Draw the label
    draw_text(win, label, label_x, label_y, FONT_SMALL, BLACK, centered=True)

    # Adjust start_y for bonus markers to be below the label
    start_y += 30  # Space below the label

    alphabatized_bonus_markers = sorted(game.selected_map.bonus_marker_pool, key=lambda bm: bm)
    max_per_column = 6
    vertical_space = 20  # Space between markers

    if len(alphabatized_bonus_markers) > max_per_column:
        # If more than 6, split into two columns
        first_six_bonus_markers = alphabatized_bonus_markers[:max_per_column]
        second_six_bonus_markers = alphabatized_bonus_markers[max_per_column:]

        # Define starting positions for each column
        column1_x = start_x + bonus_marker_pool_text_box_width // 4
        column2_x = start_x + 3 * bonus_marker_pool_text_box_width // 4
        column_y = start_y

        # Draw first column
        for bm in first_six_bonus_markers:
            draw_text(win, bm, column1_x, column_y, FONT_SMALL, BLACK, centered=True)
            column_y += vertical_space

        # Draw second column
        column_y = start_y  # Reset y for second column
        for bm in second_six_bonus_markers:
            draw_text(win, bm, column2_x, column_y, FONT_SMALL, BLACK, centered=True)
            column_y += vertical_space
    else:
        # If 6 or less, keep in a single column and center
        bm_start_x = start_x + bonus_marker_pool_text_box_width // 2
        bm_start_y = start_y
        for bm in alphabatized_bonus_markers:
            draw_text(win, bm, bm_start_x, bm_start_y, FONT_SMALL, BLACK, centered=True)
            bm_start_y += vertical_space

def redraw_window(win, game):
    selected_map = game.selected_map

    draw_bonus_markers(win, selected_map)
    draw_upgrades(win, selected_map)   
    draw_cities_and_offices(win, selected_map.cities)
    draw_routes(win, selected_map.routes)
    draw_actions_remaining(win, game)
    draw_scoreboard(win, game.players, selected_map.map_width+600, selected_map.map_height-170)
    draw_completed_cities_indicator(win, game)

    for player in game.players:
        draw_player_board(win, player, game.current_player)

    if game.current_player.actions_remaining == 0:
        if game.current_player.bonus_markers and any(bm.type != 'PlaceAdjacent' for bm in game.current_player.bonus_markers): 
            draw_end_turn(win, game)
    
    draw_get_game_state_button(win, game)
    draw_bonus_marker_pool(win, game)

def draw_player_board(window, player, current_player):
    board = player.board
    
    # Draw board background with player color
    draw_shape(window, "rectangle", board.player.color, board.x, board.y, board.width, board.height)

    draw_city_keys_section(window, board)
    draw_privilegium_section(window, board)
    draw_bonus_markers_section(window, board)
    draw_liber_sophiae_section(window, board)
    draw_actiones_section(window, board)
    draw_bank_section(window, board)
    draw_general_stock(window, board)
    draw_personal_supply(window, board)

    if player == current_player:
        draw_circle_selection_buttons(window, board)

def draw_city_keys_section(window, board):
    # Draw "City Keys" section as diamonds
    for i, value in enumerate(CITY_KEYS_MAX_VALUES):
        color = WHITE if board.player.has_unlocked_key(i) else board.player.color
        points = [
            (board.x + 10 + i * (SQUARE_SIZE + 5), board.y + 10 + SQUARE_SIZE // 2),
            (board.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, board.y + 10),
            (board.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE, board.y + 10 + SQUARE_SIZE // 2),
            (board.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, board.y + 10 + SQUARE_SIZE)
        ]
        draw_shape(window, "polygon", color, None, None, points=points)
        draw_text(window, str(value), board.x + 10 + i * (SQUARE_SIZE + 5) + SQUARE_SIZE // 2, board.y + 10 + SQUARE_SIZE // 2, FONT_PLAYERBOARD, BLACK, centered=True)
    
    draw_text(window, "Keys", board.x + 10, board.y + 10 + SQUARE_SIZE + 5, FONT_PLAYERBOARD, BLACK)

def draw_privilegium_section(window, board):
    # Adjusted "Privilegium" section with buffer moved further down
    privilege_y = board.y + 10 + 2*SQUARE_SIZE + 10  # adjusting spacing
    colors = [WHITE, ORANGE, PINK, BLACK]
    for i, color in enumerate(colors):
        if board.player.has_unlocked_privilege(i):
            color_to_use = color
        else:
            color_to_use = board.player.color
        draw_shape(window, "rectangle", color_to_use, board.x + 10 + i*(SQUARE_SIZE + 5), privilege_y, width=SQUARE_SIZE, height=SQUARE_SIZE)
    
    draw_text(window, "Privilege", board.x + 10, privilege_y + SQUARE_SIZE + 5, FONT_PLAYERBOARD, BLACK)

def draw_bonus_markers_section(window, board):
    # Starting position for bonus markers (below the privilege section)
    bm_start_y = board.y + board.height - CIRCLE_RADIUS*2  # adjusting spacing

    for i, bm in enumerate(board.player.bonus_markers):
        bm_x = board.x + 20 + CIRCLE_RADIUS + (i * (CIRCLE_RADIUS*2 + 30))
        board.player.bonus_markers[i].position = (bm_x, bm_start_y) #update the position
        draw_board_bonus_markers(window, bm, board.player.bonus_markers[i].position)

def draw_liber_sophiae_section(window, board):
        # Draw "Liber Sophiae" section (circles)
    colors = [WHITE, ORANGE, PINK, BLACK]
    board.start_x = board.x + 10 + len(colors)*50 + 5*(len(colors)-1) + 10  # buffer after Privilege

    circle_label = FONT_PLAYERBOARD.render("Liber Sophiae", True, BLACK)
    circle_section_width = len(BOOK_OF_KNOWLEDGE_MAX_VALUES) * (CIRCLE_RADIUS*2 + 5)  # total width of all circles combined
    circle_label_width = circle_label.get_width()

    # Adjust start_x to center the "Liber Sophiae" section
    board.start_x += (circle_section_width - circle_label_width) // 2

    for i, value in enumerate(BOOK_OF_KNOWLEDGE_MAX_VALUES):
        if board.player.has_unlocked_book(i):
            color = WHITE
        else:
            color = board.player.color

        draw_shape(window, "circle", color, board.start_x + i*(CIRCLE_RADIUS*2 + 5), board.y + 10 + CIRCLE_RADIUS, width=CIRCLE_RADIUS)
        draw_text(window, str(value), board.start_x + i*(CIRCLE_RADIUS*2 + 5), board.y + 10 + CIRCLE_RADIUS, FONT_PLAYERBOARD, BLACK, centered=True)

    draw_text(window, "Liber Sophiae", board.start_x-CIRCLE_RADIUS, board.y + 10 + CIRCLE_RADIUS*2 + 5, FONT_PLAYERBOARD, BLACK)
            
    board.start_x += len(BOOK_OF_KNOWLEDGE_MAX_VALUES) * (CIRCLE_RADIUS * 2 + 5) + 10

def draw_actiones_section(window, board):
    # Calculate y-position for the "Actiones" section first
    board.actions_y = board.y + 10

    # Draw "Actiones" section (squares)
    for i, value in enumerate(ACTIONS_MAX_VALUES):
        # Color the boxes up to actions_index with WHITE and the rest with player's color
        color = WHITE if i <= board.player.actions_index else board.player.color

        draw_shape(window, "rectangle", color, board.start_x + i*(SQUARE_SIZE + 5), board.actions_y, width=SQUARE_SIZE, height=SQUARE_SIZE)
        draw_text(window, str(value), board.start_x + i*(SQUARE_SIZE + 5) + SQUARE_SIZE // 2, board.actions_y + SQUARE_SIZE // 2, FONT_PLAYERBOARD, BLACK, centered=True)

    draw_text(window, "Actiones", board.start_x, board.actions_y + SQUARE_SIZE + 5, FONT_PLAYERBOARD, BLACK)

def draw_bank_section(window, board):
    # Calculate y-position for the "Bank" section based on "Actiones" section height
    bank_y = board.actions_y + SQUARE_SIZE + 5 + FONT_PLAYERBOARD.get_height() + 5

    # Draw "Bank" section (squares)
    for i, value in enumerate(BANK_MAX_VALUES):
        if board.player.has_unlocked_bank(i):
            color = WHITE
        else:
            color = board.player.color

        draw_shape(window, "rectangle", color, board.start_x + i*(SQUARE_SIZE + 5), bank_y, width=SQUARE_SIZE, height=SQUARE_SIZE)
        draw_text(window, str(value), board.start_x + i*(SQUARE_SIZE + 5) + SQUARE_SIZE // 2, bank_y + SQUARE_SIZE // 2, FONT_PLAYERBOARD, BLACK, centered=True)

    draw_text(window, "Bank", board.start_x, bank_y + SQUARE_SIZE + 5, FONT_PLAYERBOARD, BLACK)

def draw_general_stock(window, board):
    x_offset = board.x + 650  # Adjust this value based on exact positioning
    y_offset = board.y + 10  # Adjust this value based on exact positioning

    # Draw 'GS:' text
    font = pygame.font.SysFont(None, 36)
    draw_text(window, 'GS:', x_offset, y_offset, font, BLACK)
    
    # Draw square for squares count with the number inside
    x_offset += 50
    draw_shape(window, 'rectangle', board.player.color, x_offset, y_offset, 40, 40)
    draw_text(window, str(board.player.general_stock_squares), x_offset + 20, y_offset + 20, font, (0, 0, 0), centered=True)  # Assuming black color for numbers

    # Draw circle next to square for circles count with number inside
    x_offset += 60
    draw_shape(window, 'circle', board.player.color, x_offset, y_offset + 20, 20) # Assuming radius is half of square side
    draw_text(window, str(board.player.general_stock_circles), x_offset, y_offset + 20, font, (0, 0, 0), centered=True)  # Assuming black color for numbers
    
def draw_personal_supply(window, board):
    x_offset = board.x + 650  # Adjust this value based on exact positioning
    y_offset = board.y + 80  # Positioning it further down than GS for clarity

    # Draw 'PS:' text
    font = pygame.font.SysFont(None, 36)
    draw_text(window, 'PS:', x_offset, y_offset, font, BLACK)

    # Draw square for squares count in personal supply with the number inside
    x_offset += 50
    draw_shape(window, 'rectangle', board.player.color, x_offset, y_offset, 40, 40)
    draw_text(window, str(board.player.personal_supply_squares), x_offset + 20, y_offset + 20, font, (0, 0, 0), centered=True)

    # Draw circle next to square for circles count in personal supply with number inside
    x_offset += 60
    draw_shape(window, 'circle', board.player.color, x_offset, y_offset + 20, 20)
    draw_text(window, str(board.player.personal_supply_circles), x_offset, y_offset + 20, font, (0, 0, 0), centered=True)

def draw_circle_selection_buttons(window, board):
    # Starting position for the Income label
    income_x = board.x + board.width - 220
    income_y = board.y + board.height - 70

    # Button dimensions and spacing
    button_width = 60
    button_height = 30
    horizontal_spacing = 70  # Spacing between columns
    vertical_spacing = 35    # Spacing between rows

    # Clear any previously drawn buttons
    board.circle_buttons = []
    
    # Use the `draw_text` function for centering on both X and Y axis
    font = pygame.font.SysFont(None, 28)

    # Calculate the center coordinates of the button
    button_center_x = income_x + button_width / 2
    button_center_y = income_y + button_height / 2

    # Draw the Income text centered on both axes
    draw_text(window, "Income:", button_center_x, button_center_y, font, BLACK, centered=True)

    # Logic for determining which buttons to display based on the conditions
    board.button_labels = board.player.income_action_based_on_circle_count(min(board.player.general_stock_circles, 4), board.player.bank, board.player.general_stock_squares)

    if len(board.button_labels) == 1 and board.player.bank == 50:  #max bank
        button_x = income_x + horizontal_spacing
        button_y = income_y + vertical_spacing / 2
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        board.circle_buttons.append(button_rect)
        draw_shape(window, 'rectangle', BLACK, button_x, button_y, button_width, button_height)
        draw_text(window, board.button_labels[0], button_x + (button_width // 2), button_y + (button_height // 2), pygame.font.SysFont(None, 24), WHITE, centered=True)
        return
    
    for i, label in enumerate(board.button_labels):
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
        board.circle_buttons.append(button_rect)
        draw_shape(window, 'rectangle', BLACK, button_x, button_y, button_width, button_height)

        # Label for the button
        draw_text(window, label, button_x + (button_width // 2), button_y + (button_height // 2), pygame.font.SysFont(None, 24), WHITE, centered=True)