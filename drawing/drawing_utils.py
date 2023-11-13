import pygame
from map_data.constants import WHITE, TAN, COLOR_NAMES, BLACK, BUFFER, SQUARE_SIZE, SPACING, CIRCLE_RADIUS

pygame.init()

FONT_LARGE = pygame.font.Font(None, 36)  # Initialize once and use everywhere
FONT_SMALL = pygame.font.Font(None, 24)  # Initialize once and use everywhere
BORDER_WIDTH = 2 #black outline of

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
    
    """
    Draws text on the screen.

    Parameters:
    - window: the pygame surface where the text will be drawn.
    - text: the string of text to display.
    - x, y: the coordinates where the text will be placed.
    - font: a pygame font object.
    - color: the color of the text. Default is BLACK.
    - centered: a flag to determine if the text should be centered on the x, y coords. Default is False.
    """
def draw_text(window, text, x, y, font, color=BLACK, centered=False):
    text_surface = font.render(text, True, color)

    if centered:
        text_rect = text_surface.get_rect(center=(x, y))
    else:
        text_rect = text_surface.get_rect(topleft=(x, y))

    window.blit(text_surface, text_rect)

def draw_upgrades(win, selected_map):
    for upgrade_types in selected_map.upgrade_cities:
        upgrade_types.draw_upgrades_on_map(win)
    selected_map.specialprestigepoints.draw_special_prestige_points(win)
    
def draw_completed_cities_indicator(win, selected_map):
    full_cities = [city for city in selected_map.cities if city.city_is_full()]
    num_full_cities = len(full_cities)
    font_size = 24  # Define the font size for the numbers
    label_font_size = 20  # Define the font size for the label

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
        draw_line(win, WHITE, route.cities[0].midpoint, route.cities[1].midpoint, 10, 2)
        # Check if the route has a bonus marker and call its draw method
        if route.bonus_marker:
            # Construct the key for the dictionary
            city_pair = tuple(sorted([route.cities[0].name, route.cities[1].name]))
            # Fetch the bonus marker position from the dictionary
            bonus_marker_pos = selected_map.bonus_marker_positions.get(city_pair)

            # If the position exists, call the draw method on the bonus marker
            if bonus_marker_pos:
                route.bonus_marker.draw_board_bonus_markers(win, bonus_marker_pos)
                # print(f"Drew bonus marker between {city_pair[0]} and {city_pair[1]} at position {bonus_marker_pos}")
            # else:
                # print(f"No bonus marker position found for route between {city_pair[0]} and {city_pair[1]}")

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

def redraw_window(win, game):
    selected_map = game.selected_map

    draw_bonus_markers(win, selected_map)
    draw_upgrades(win, selected_map)   
    draw_cities_and_offices(win, selected_map.cities)
    draw_routes(win, selected_map.routes)
    draw_actions_remaining(win, game)
    draw_scoreboard(win, game.players, selected_map.map_width+600, selected_map.map_height-170)
    draw_completed_cities_indicator(win, selected_map)

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

def draw_end_game(win, winning_player):
    # Create a semi-transparent surface
    transparent_surface = pygame.Surface(win.get_size(), pygame.SRCALPHA)
    transparent_surface.fill((255, 255, 255, 128))  # 128 here for 50% transparency

    # Blit the semi-transparent surface onto the window
    win.blit(transparent_surface, (0, 0))

    winner_text = FONT_LARGE.render(f"Game Over! {COLOR_NAMES[winning_player.color]} wins!", True, winning_player.color)
    rect = winner_text.get_rect(center=win.get_rect().center)
    win.blit(winner_text, rect)
    pygame.display.update()