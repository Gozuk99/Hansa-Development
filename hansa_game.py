"""Primary interactive entry point for Hansa Teutonica."""

from drawing.game_window import GameWindow
from drawing.new_game_menu import run_new_game_menu


def main() -> int:
    configuration = run_new_game_menu()
    if configuration is None:
        return 0
    game = configuration.create_game()
    GameWindow(game).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
