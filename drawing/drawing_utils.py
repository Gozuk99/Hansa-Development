import pygame
from map_data.constants import BLACK

BORDER_WIDTH = 2 #black outline of shapes

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