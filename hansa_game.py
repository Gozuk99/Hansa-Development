"""Primary interactive entry point for Hansa Teutonica."""


def main() -> int:
    import pygame

    pygame.init()
    try:
        from drawing.game_window import GameWindow
        from drawing.new_game_menu import run_new_game_menu
        from game.game_info import Game

        selection = run_new_game_menu()
        if selection is None:
            return 0
        game = selection if isinstance(selection, Game) else selection.create_game()
        GameWindow(game).run()
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
