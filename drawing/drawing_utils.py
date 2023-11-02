import pygame
from map_data.constants import WHITE, TAN, COLOR_NAMES, BLACK, BUFFER, SQUARE_SIZE, SPACING, CIRCLE_RADIUS

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
def draw_line(surface, color, start_pos, end_pos, line_width, border_width):
    # Draw the thicker border line first
    pygame.draw.line(surface, BLACK, start_pos, end_pos, line_width + 2 * border_width)
    # Draw the main line on top of the border
    pygame.draw.line(surface, color, start_pos, end_pos, line_width)

last_text_box = None
def redraw_window(win, cities, routes, current_player, waiting_for_displaced_player, displaced_player, WIDTH, HEIGHT):
    global last_text_box
    # Draw cities and their offices
    font = pygame.font.Font(None, 36)

    # Clear the area with the background color
    if last_text_box is not None:
        pygame.draw.rect(win, TAN, last_text_box)

    for city in cities:
        # Calculate the position of the rectangle
        rect_x = city.pos[0]
        rect_y = city.pos[1]

        # Use draw_shape function to draw the city rectangle with a border
        draw_shape(win, "rectangle", city.color, rect_x, rect_y, city.width, city.height)
        # Calculate text position to place it just below the city rectangle
        text_width = font.size(city.name)[0]
        text_x = city.pos[0] + (city.width - text_width) // 2
        text_y = city.pos[1] + city.height

        # Use draw_text function to render the city name below the rectangle
        draw_text(win, city.name, text_x, text_y, font, BLACK)

        start_x = rect_x + BUFFER  # Starting x-coordinate within the rectangle
        start_y = rect_y + city.height // 2 - SQUARE_SIZE // 2  # Centered vertically in the rectangle

        for office in city.offices:
            if office.shape == "square":
                draw_shape(win, "rectangle", office.color, start_x, start_y, SQUARE_SIZE, SQUARE_SIZE)
                start_x += SQUARE_SIZE + SPACING

            else:  
                circle_x = start_x + CIRCLE_RADIUS
                circle_y = start_y + SQUARE_SIZE // 2
                draw_shape(win, "circle", office.color, circle_x, circle_y, CIRCLE_RADIUS)
                start_x += CIRCLE_RADIUS * 2 + SPACING

    for route in routes:
        for post in route.posts:
            post_x, post_y = post.pos

            # Draw the circle
            draw_shape(win, "circle", post.circle_color, post_x, post_y, width=CIRCLE_RADIUS)

            # Draw the square if there is no owner or if the owner has placed a square piece
            if post.owner_piece_shape is None or post.owner_piece_shape == "square":
                draw_shape(win, "rectangle", post.square_color, post_x - SQUARE_SIZE // 2, post_y - SQUARE_SIZE // 2, width=SQUARE_SIZE, height=SQUARE_SIZE)
   
    if waiting_for_displaced_player:
        text_str = f"{COLOR_NAMES[current_player.color]} displaced {COLOR_NAMES[displaced_player.player.color]} - waiting for {COLOR_NAMES[displaced_player.player.color]} to place {displaced_player.total_pieces_to_place} pieces!"
        combined_text = font.render(text_str, True, COLOR_NAMES[current_player.color])

        # Determine the position and size of the text area
        text_width = combined_text.get_width()
        text_height = combined_text.get_height()
        text_area = pygame.Rect(WIDTH // 2 - text_width // 2, HEIGHT - 50 - text_height // 2, text_width, text_height)
        last_text_box = text_area
    else:
        text_str = f"Actions: {current_player.actions_remaining}"
        combined_text = font.render(text_str, True, COLOR_NAMES[current_player.color])

        # Determine the position and size of the text area
        text_width = combined_text.get_width()
        text_height = combined_text.get_height()
        text_area = pygame.Rect(WIDTH // 2 - text_width // 2, HEIGHT - 50 - text_height // 2, text_width, text_height)
        last_text_box = text_area

    # Clear the area with the background color
    pygame.draw.rect(win, WHITE, text_area)

    # Draw the combined text onto the main window
    win.blit(combined_text, (WIDTH // 2 - text_width // 2, HEIGHT - 50 - text_height // 2))
    pygame.display.update(text_area)

def draw_end_game(win, winning_player):
    font = pygame.font.Font(None, 36)
    win.fill((255, 255, 255, 80))  # semi-transparent white background
    winner_text = font.render(f"Game Over! {COLOR_NAMES[winning_player.color]} wins!", True, winning_player.color)
    rect = winner_text.get_rect()
    rect.center = win.get_rect().center
    win.blit(winner_text, rect)
    pygame.display.update()