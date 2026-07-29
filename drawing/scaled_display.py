"""A resizable Pygame window that presents a fixed-size logical canvas."""

from __future__ import annotations

import pygame


class ScaledDisplay:
    """Keep drawing coordinates stable while fitting the whole UI on screen."""

    def __init__(self, logical_size: tuple[int, int], caption: str):
        self.logical_size = logical_size
        self.canvas = pygame.Surface(logical_size)
        pygame.display.set_caption(caption)
        info = pygame.display.Info()
        available = (
            max(640, info.current_w - 80) if info.current_w else logical_size[0],
            max(480, info.current_h - 100) if info.current_h else logical_size[1],
        )
        self.window = pygame.display.set_mode(
            self.fit_size(logical_size, available),
            pygame.RESIZABLE,
        )

    @staticmethod
    def fit_size(
        logical_size: tuple[int, int],
        available_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Return the largest aspect-preserving size within the available area."""
        logical_width, logical_height = logical_size
        scale = min(
            1.0,
            available_size[0] / logical_width,
            available_size[1] / logical_height,
        )
        return max(1, round(logical_width * scale)), max(1, round(logical_height * scale))

    def resize(self, size: tuple[int, int]) -> None:
        self.window = pygame.display.set_mode(
            (max(320, size[0]), max(240, size[1])),
            pygame.RESIZABLE,
        )

    def to_logical(self, position: tuple[int, int]) -> tuple[int, int]:
        target = self.presentation_rect()
        return (
            round((position[0] - target.x) * self.logical_size[0] / target.width),
            round((position[1] - target.y) * self.logical_size[1] / target.height),
        )

    def presentation_rect(self) -> pygame.Rect:
        target_size = self.fit_size(self.logical_size, self.window.get_size())
        return pygame.Rect(
            (self.window.get_width() - target_size[0]) // 2,
            (self.window.get_height() - target_size[1]) // 2,
            *target_size,
        )

    def present(self) -> None:
        target = self.presentation_rect()
        frame = pygame.transform.smoothscale(self.canvas, target.size)
        self.window.fill((20, 20, 20))
        self.window.blit(frame, target)
        pygame.display.flip()
