import pygame

# Initialize Pygame
pygame.init()

# Set up constants
CIRCLE_RADIUS = 20
SQUARE_SIZE = 26
BUFFER = 10  # Buffer around the rectangle
SHAPES = 5  # Total number of shapes (circles and squares)
SPACING = 0  # No spacing between the shapes

# Calculate the width of the rectangle with buffer at both ends
RECT_WIDTH = (
    CIRCLE_RADIUS * 2 +  # Width of the first shape
    (SHAPES - 1) * (CIRCLE_RADIUS * 2) +  # Total width of all shapes except the first
    3 * BUFFER  # Buffer at the beginning and end
)
RECT_HEIGHT = max(CIRCLE_RADIUS * 2, SQUARE_SIZE) + 2 * BUFFER

# Create a Pygame window
win = pygame.display.set_mode((600, 600))  # Adjust window size as needed
pygame.display.set_caption("Shapes with Buffer")

# Set colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Calculate the position of the rectangle
rect_x = 200 - RECT_WIDTH // 2
rect_y = 200 - RECT_HEIGHT // 2

# Calculate the spacing between shapes
spacing = 2

# Calculate the positions of the shapes within the rectangle
shape_positions = []

for i in range(SHAPES):
    # Calculate the x-coordinate of the shape, considering the buffer for the first shape
    start_location = rect_x + BUFFER + CIRCLE_RADIUS
    if i == 0:
        shape_x = start_location
    else:
        shape_x = start_location + i * (CIRCLE_RADIUS * 2 + spacing)
    shape_y = rect_y + RECT_HEIGHT // 2  # Center vertically in the rectangle
    shape_positions.append((shape_x, shape_y))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    win.fill(WHITE)  # Fill the window with white

    # Draw the rectangle with buffer
    pygame.draw.rect(win, BLUE, (rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT))

    # Draw the circles and squares
    for i, (shape_x, shape_y) in enumerate(shape_positions):
        if i % 2 == 0:
            pygame.draw.circle(win, RED, (shape_x, shape_y), CIRCLE_RADIUS)
        else:
            pygame.draw.rect(win, GREEN, (shape_x - SQUARE_SIZE / 2, shape_y - SQUARE_SIZE / 2, SQUARE_SIZE, SQUARE_SIZE))

    pygame.display.update()

pygame.quit()



