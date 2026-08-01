"""Pygame new-game menu backed by :class:`GameConfiguration`."""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

from game.game_config import (
    EMPERORS_FAVOUR_TILES,
    GameConfiguration,
    PlayerControl,
    human_players,
)
from game.setup import MAX_PLAYERS, MIN_PLAYERS, SUPPORTED_MAPS
from drawing.scaled_display import ScaledDisplay
from drawing.save_dialogs import choose_load_file
from game.persistence import load_game
from map_data.map_attributes import Map


WINDOW_SIZE = (980, 940)
INITIAL_WINDOW_SCALE = 2.0
BACKGROUND = (237, 222, 190)
PANEL = (250, 243, 224)
INK = (38, 31, 24)
ACCENT = (117, 70, 42)
ACTIVE = (72, 121, 86)
ERROR = (160, 45, 45)
INACTIVE = (155, 151, 143)

BONUS_MARKER_OPTIONS = {
    **Map.STANDARD_BONUS_MARKER_SUPPLY,
    **Map.PROMO_BONUS_MARKERS,
}


@dataclass
class NewGameMenuState:
    map_num: int = 1
    player_count: int = 3
    player_controls: list[PlayerControl] = field(default_factory=lambda: list(human_players(3)))
    use_mission_cards: bool = False
    use_emperors_favour: bool = False
    emperor_tile_mode: str = "random"
    emperor_tiles: list[str] = field(default_factory=list)
    use_promo_markers: bool = False
    promo_marker_mode: str = "random"
    promo_markers: list[str] = field(default_factory=list)
    seed: int | None = None

    def set_player_count(self, count: int) -> None:
        self.player_count = count
        self.player_controls = self.player_controls[:count]
        self.player_controls.extend([PlayerControl.HUMAN] * (count - len(self.player_controls)))
        if len(self.emperor_tiles) != count:
            self.emperor_tiles.clear()

    def set_map(self, map_num: int) -> None:
        self.map_num = map_num
        if map_num != 1:
            self.use_mission_cards = False

    def build_configuration(self) -> GameConfiguration:
        return GameConfiguration(
            map_num=self.map_num,
            player_count=self.player_count,
            player_controls=tuple(self.player_controls),
            use_mission_cards=self.use_mission_cards,
            use_emperors_favour=self.use_emperors_favour,
            emperor_tile_mode=self.emperor_tile_mode,
            emperor_tiles=tuple(self.emperor_tiles),
            use_promo_markers=self.use_promo_markers,
            promo_marker_mode=self.promo_marker_mode,
            promo_markers=tuple(self.promo_markers),
            seed=self.seed,
        )


class NewGameMenu:
    def __init__(self, state: NewGameMenuState | None = None):
        self.state = state or NewGameMenuState()
        self.display = ScaledDisplay(
            WINDOW_SIZE,
            "Hansa Teutonica — New Game",
            initial_scale=INITIAL_WINDOW_SCALE,
        )
        self.screen = self.display.canvas
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 44)
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.buttons: list[tuple[pygame.Rect, object]] = []
        self.error = ""

    def button(self, rect, text, action, selected=False, small=False, enabled=True):
        color = ACTIVE if selected else ACCENT
        if not enabled:
            color = INACTIVE
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        font = self.small_font if small else self.font
        label = font.render(text, True, (255, 255, 255))
        self.screen.blit(label, label.get_rect(center=rect.center))
        if enabled:
            self.buttons.append((rect, action))

    def label(self, text, x, y, *, small=False, color=INK):
        font = self.small_font if small else self.font
        self.screen.blit(font.render(text, True, color), (x, y))

    def draw(self):
        self.buttons.clear()
        self.screen.fill(BACKGROUND)
        pygame.draw.rect(self.screen, PANEL, (24, 20, 932, 890), border_radius=12)
        self.screen.blit(
            self.title_font.render("Create New Game", True, INK),
            (48, 38),
        )

        self.label("Players", 48, 100)
        for offset, count in enumerate(range(MIN_PLAYERS, MAX_PLAYERS + 1)):
            self.button(
                pygame.Rect(180 + offset * 68, 94, 58, 34),
                str(count),
                lambda value=count: self.state.set_player_count(value),
                selected=self.state.player_count == count,
            )

        self.label("Map", 500, 100)
        for offset, map_num in enumerate(SUPPORTED_MAPS):
            self.button(
                pygame.Rect(580 + offset * 68, 94, 58, 34),
                str(map_num),
                lambda value=map_num: self.state.set_map(value),
                selected=self.state.map_num == map_num,
            )

        self.label("Player controllers (all seats default to Human)", 48, 155)
        controls = tuple(PlayerControl)
        for index in range(MAX_PLAYERS):
            active_seat = index < self.state.player_count
            y = 188 + index * 39
            self.label(f"Player {index + 1}", 55, y + 8, small=True)
            for control_index, control in enumerate(controls):
                width = 138
                x = 145 + control_index * 148
                self.button(
                    pygame.Rect(x, y, width, 36),
                    "Magnus" if control is PlayerControl.MAGNUS else control.value,
                    lambda seat=index, value=control: self._set_control(seat, value),
                    selected=(active_seat and self.state.player_controls[index] is control),
                    small=True,
                    enabled=active_seat,
                )

        y = 398
        if self.state.map_num == 1:
            self.label("Mission Cards", 48, y + 7)
            self.button(
                pygame.Rect(270, y, 100, 36),
                "Enabled" if self.state.use_mission_cards else "Disabled",
                self._toggle_missions,
                selected=self.state.use_mission_cards,
                small=True,
            )
            y += 50

        self.label("Emperor's Favour", 48, y + 7)
        self.button(
            pygame.Rect(270, y, 100, 36),
            "Enabled" if self.state.use_emperors_favour else "Disabled",
            self._toggle_emperor,
            selected=self.state.use_emperors_favour,
            small=True,
        )
        if self.state.use_emperors_favour:
            self._draw_mode_buttons(y, "emperor", x=390)
            y += 48
            if self.state.emperor_tile_mode == "manual":
                self.label(
                    f"Choose exactly {self.state.player_count} tiles:",
                    70,
                    y,
                    small=True,
                )
                y += 28
                for index, tile in enumerate(EMPERORS_FAVOUR_TILES):
                    x = 70 + (index % 3) * 285
                    row_y = y + (index // 3) * 38
                    self.button(
                        pygame.Rect(x, row_y, 270, 31),
                        tile,
                        lambda value=tile: self._toggle_tile(value),
                        selected=tile in self.state.emperor_tiles,
                        small=True,
                    )
                y += 82
        y += 48

        self.label("Custom Bonus-Marker Supply", 48, y + 7)
        self.button(
            pygame.Rect(350, y, 100, 36),
            "Enabled" if self.state.use_promo_markers else "Disabled",
            self._toggle_promos,
            selected=self.state.use_promo_markers,
            small=True,
        )
        if self.state.use_promo_markers:
            self._draw_mode_buttons(y, "promo", x=470)
            y += 48
            if self.state.promo_marker_mode == "manual":
                self.label(
                    f"Choose exactly 12 markers ({len(self.state.promo_markers)}/12): "
                    "left-click to add; right-click to remove",
                    70,
                    y,
                    small=True,
                )
                y += 28
                for index, (marker, maximum) in enumerate(BONUS_MARKER_OPTIONS.items()):
                    count = self.state.promo_markers.count(marker)
                    self.button(
                        pygame.Rect(
                            70 + (index % 3) * 285,
                            y + (index // 3) * 36,
                            270,
                            32,
                        ),
                        f"{marker}: {count}/{maximum}",
                        ("promo_count", marker),
                        selected=count > 0,
                        small=True,
                    )
                y += 113

        if self.error:
            self.label(self.error, 48, 824, small=True, color=ERROR)
        self.button(
            pygame.Rect(710, 856, 210, 44),
            "Start Game",
            self._start,
            selected=True,
        )
        self.button(
            pygame.Rect(480, 856, 210, 44),
            "Load Saved Game",
            self._load_saved_game,
        )
        self.display.present()

    def _draw_mode_buttons(self, y, module, x=330):
        mode = self.state.emperor_tile_mode if module == "emperor" else self.state.promo_marker_mode
        for offset, value in enumerate(("random", "manual")):
            self.button(
                pygame.Rect(x + offset * 105, y, 95, 36),
                value.title(),
                lambda selected=value, target=module: self._set_mode(target, selected),
                selected=mode == value,
                small=True,
            )

    def _set_control(self, index, control):
        if index < self.state.player_count:
            self.state.player_controls[index] = control

    def _toggle_missions(self):
        self.state.use_mission_cards = not self.state.use_mission_cards

    def _toggle_emperor(self):
        self.state.use_emperors_favour = not self.state.use_emperors_favour
        if not self.state.use_emperors_favour:
            self.state.emperor_tiles.clear()

    def _toggle_promos(self):
        self.state.use_promo_markers = not self.state.use_promo_markers
        if not self.state.use_promo_markers:
            self.state.promo_markers.clear()

    def _set_mode(self, module, mode):
        if module == "emperor":
            self.state.emperor_tile_mode = mode
            self.state.emperor_tiles.clear()
        else:
            self.state.promo_marker_mode = mode
            self.state.promo_markers.clear()

    def _toggle_tile(self, tile):
        if tile in self.state.emperor_tiles:
            self.state.emperor_tiles.remove(tile)
        elif len(self.state.emperor_tiles) < self.state.player_count:
            self.state.emperor_tiles.append(tile)

    def _change_promo_count(self, marker, direction):
        count = self.state.promo_markers.count(marker)
        maximum = BONUS_MARKER_OPTIONS[marker]
        if direction > 0 and count < maximum and len(self.state.promo_markers) < 12:
            self.state.promo_markers.append(marker)
        elif direction < 0 and count:
            self.state.promo_markers.remove(marker)

    def _start(self):
        try:
            self.result = self.state.build_configuration()
        except ValueError as error:
            self.error = str(error)
        else:
            self.running = False

    def _load_saved_game(self):
        try:
            filename = choose_load_file()
            if filename is None:
                return
            self.result = load_game(filename)
        except Exception as error:
            self.error = str(error)
        else:
            self.running = False

    def run(self) -> object | None:
        self.result = None
        self.running = True
        while self.running:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.VIDEORESIZE:
                    self.display.resize(event.size)
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self._start()
                    elif event.key == pygame.K_ESCAPE:
                        return None
                if event.type == pygame.MOUSEBUTTONUP:
                    logical_position = self.display.to_logical(event.pos)
                    for rect, action in reversed(self.buttons):
                        if not rect.collidepoint(logical_position):
                            continue
                        if isinstance(action, tuple) and action[0] == "promo_count":
                            self._change_promo_count(
                                action[1],
                                -1 if event.button == 3 else 1,
                            )
                        else:
                            action()
                        break
            self.clock.tick(30)
        return self.result


def run_new_game_menu() -> object | None:
    pygame.init()
    return NewGameMenu().run()
