
CIRCLE_RADIUS = 20
SQUARE_SIZE = 26
BUFFER = 10  # Buffer around the rectangle
SPACING = 10  # Increased spacing between the shapes

# Constants for input and output sizes
INPUT_SIZE = 2314
OUTPUT_SIZE = 616

# Alphabetized Color variables
BLACK = (0, 0, 0)
BLACKISH_BROWN = (130, 100, 80)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 105, 148)
DARK_GREEN = (34, 139, 34)
DARK_RED = (139, 0, 0)
GREEN = (0, 255, 0)
GREY = (128, 128, 128)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
PURPLE = (128, 0, 128)
RED = (255, 0, 0)
TAN = (210, 180, 140)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)

COLOR_NAMES = {
    BLACK: "BLACK",
    BLACKISH_BROWN: "BLACKISH_BROWN",
    BLUE: "BLUE",
    DARK_BLUE : "DARK_BLUE",
    DARK_GREEN : "DARK_GREEN",
    DARK_RED : "DARK_RED",
    GREEN: "GREEN",
    GREY: "GREY",
    ORANGE: "ORANGE",
    PINK: "PINK",
    PURPLE: "PURPLE",
    RED: "RED",
    TAN: "TAN",
    WHITE: "WHITE",
    YELLOW: "YELLOW"
}

CITY_KEYS_MAX_VALUES = [1, 2, 2, 3, 4]
PRIVILEGE_COLORS = ["WHITE", "ORANGE", "PINK", "BLACK"]
BOOK_OF_KNOWLEDGE_MAX_VALUES = [2, 3, 4, 5]
ACTIONS_MAX_VALUES = [2, 3, 3, 4, 4, 5]
BANK_MAX_VALUES = [3, 5, 7, 50]

UPGRADE_METHODS_MAP = {
    "keys": "upgrade_keys",
    "actions": "upgrade_actions",
    "privilege": "upgrade_privilege",
    "book": "upgrade_book",
    "bank": "upgrade_bank"
}
UPGRADE_MAX_VALUES = {
    'keys': CITY_KEYS_MAX_VALUES[-1],
    'privilege': PRIVILEGE_COLORS[-1],
    'book': BOOK_OF_KNOWLEDGE_MAX_VALUES[-1],
    'actions': ACTIONS_MAX_VALUES[-1],
    'bank': BANK_MAX_VALUES[-1]
}

