"""Primary interactive entry point for Hansa Teutonica."""


def main() -> int:
    import pygame

    pygame.init()
    try:
        from drawing.game_window import GameWindow
        from drawing.new_game_menu import run_new_game_menu

        configuration = run_new_game_menu()
        if configuration is None:
            return 0
        game = configuration.create_game()
        GameWindow(game).run()
        return 0
    finally:
        pygame.quit()


if __name__ == "__main__":
    raise SystemExit(main())
