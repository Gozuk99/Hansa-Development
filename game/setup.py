from dataclasses import dataclass


MIN_PLAYERS = 3
MAX_PLAYERS = 5
SUPPORTED_MAPS = (1, 2, 3)


@dataclass(frozen=True)
class StartingInventory:
    personal_supply_squares: int
    personal_supply_circles: int
    general_stock_squares: int
    general_stock_circles: int = 0

    @property
    def total_squares(self):
        return self.personal_supply_squares + self.general_stock_squares

    @property
    def total_circles(self):
        return self.personal_supply_circles + self.general_stock_circles


def validate_game_configuration(map_num, num_players):
    if map_num not in SUPPORTED_MAPS:
        raise ValueError(f"Unsupported map number: {map_num}")
    if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
        raise ValueError(
            f"Player count must be between {MIN_PLAYERS} and {MAX_PLAYERS}, "
            f"got {num_players}"
        )


def starting_inventory(player_order):
    if not 1 <= player_order <= MAX_PLAYERS:
        raise ValueError(f"Player order must be between 1 and {MAX_PLAYERS}")

    return StartingInventory(
        personal_supply_squares=4 + player_order,
        personal_supply_circles=1,
        general_stock_squares=7 - player_order,
    )
