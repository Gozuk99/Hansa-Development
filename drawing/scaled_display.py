"""A resizable Pygame window that presents a fixed-size logical canvas."""

from __future__ import annotations

import pygame


class ScaledDisplay:
    """Keep drawing coordinates stable while fitting the whole UI on screen."""

    def __init__(
        self,
        logical_size: tuple[int, int],
        caption: str,
        *,
        initial_scale: float = 1.0,
    ):
        self.logical_size = logical_size
        self.canvas = pygame.Surface(logical_size)
        pygame.display.set_caption(caption)
        info = pygame.display.Info()
        desktop_sizes = pygame.display.get_desktop_sizes()
        desktop_size = desktop_sizes[0] if desktop_sizes else (info.current_w, info.current_h)
        available = self.available_size(
            desktop_size,
            logical_size,
        )
        requested = (
            round(logical_size[0] * initial_scale),
            round(logical_size[1] * initial_scale),
        )
        self.window = pygame.display.set_mode(
            self.fit_size(requested, available),
            pygame.RESIZABLE,
        )

    @staticmethod
    def available_size(
        display_size: tuple[int, int],
        fallback_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Return desktop bounds with a small margin, even on tiny displays."""
        width, height = display_size
        return (
            max(1, width - 80) if width else fallback_size[0],
            max(1, height - 100) if height else fallback_size[1],
        )

    @staticmethod
    def fit_size(
        logical_size: tuple[int, int],
        available_size: tuple[int, int],
        *,
        allow_upscale: bool = False,
    ) -> tuple[int, int]:
        """Return the largest aspect-preserving size within the available area."""
        logical_width, logical_height = logical_size
        scale = min(
            available_size[0] / logical_width,
            available_size[1] / logical_height,
        )
        if not allow_upscale:
            scale = min(1.0, scale)
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
        target_size = self.fit_size(
            self.logical_size,
            self.window.get_size(),
            allow_upscale=True,
        )
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
