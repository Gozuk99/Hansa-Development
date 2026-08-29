"""Minimal legal-action-driven Pygame window for an initialized game."""

from __future__ import annotations

import random

import pygame
import torch

from ai.observation_encoder import ObservationEncoder
from drawing.action_ui import action_label, fit_text, phase_prompt
from drawing.ai_observation import public_game_state
from drawing.drawing_utils import draw_end_game, redraw_window
from drawing.scaled_display import ScaledDisplay
from drawing.save_dialogs import choose_save_file
from game.game_config import PlayerControl, choose_ranked_ai_action
from game.action_codec import DEFAULT_ACTION_CODEC
from game.action_schema import (
    ROUTE_OFFICE_SLOT_START,
    ROUTE_OUTCOME_SLOT_START,
    TILE_SLOT_BY_TYPE,
)
from game.persistence import save_game
from game.structured_actions import (
    ControlInteraction,
    PieceShape,
    PostInteraction,
    RouteInteraction,
    TileInteraction,
)
from map_data.constants import TAN


class GameWindow:
    def __init__(self, game):
        self.game = game
        self.width = game.selected_map.map_width + 1100
        self.height = game.selected_map.map_height
        self.display = ScaledDisplay((self.width, self.height), "Hansa Teutonica")
        self.screen = self.display.canvas
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.selected = 0
        self.observation_encoder = ObservationEncoder()
        self.rng = random.Random()
        saved_rng_state = getattr(game, "_saved_controller_rng_state", None)
        if saved_rng_state is None:
            self.rng.seed(game.seed)
        else:
            self.rng.setstate(saved_rng_state)
            del game._saved_controller_rng_state
        self.action_rects: list[tuple[pygame.Rect, int]] = []
        self.save_rect = pygame.Rect(0, 0, 0, 0)
        self.save_status = ""
        self.layout = None

    def legal_actions(self) -> list[int]:
        return [index for index, enabled in enumerate(self.game.ai_action_mask()) if enabled]

    def draw_action_browser(self, actions):
        self.action_rects.clear()
        panel = pygame.Rect(self.game.selected_map.map_width + 810, 10, 280, 440)
        pygame.draw.rect(self.screen, (245, 238, 218), panel)
        pygame.draw.rect(self.screen, (45, 38, 30), panel, 2)
        control = getattr(self.acting_player, "control", PlayerControl.HUMAN)
        heading = self.font.render(
            f"Player {self.acting_player.order}: {control.value}",
            True,
            (30, 25, 20),
        )
        self.screen.blit(heading, (panel.x + 10, panel.y + 10))
        help_text = self.font.render("Up/Down + Enter; E ends when legal", True, (30, 25, 20))
        self.screen.blit(help_text, (panel.x + 10, panel.y + 34))
        prompt_text = fit_text(self.font, phase_prompt(self.game), panel.width - 20)
        prompt = self.font.render(prompt_text, True, (117, 70, 42))
        self.screen.blit(prompt, (panel.x + 10, panel.y + 54))
        city_help = self.font.render(
            "City — L: office | R: complete route, no office",
            True,
            (30, 25, 20),
        )
        self.screen.blit(city_help, (panel.x + 10, panel.y + 74))
        self.save_rect = pygame.Rect(panel.x + 10, panel.bottom - 38, 120, 28)
        pygame.draw.rect(self.screen, (117, 70, 42), self.save_rect, border_radius=5)
        save_label = self.font.render("Save Game", True, (255, 255, 255))
        self.screen.blit(save_label, save_label.get_rect(center=self.save_rect.center))
        if self.save_status:
            status = fit_text(self.font, self.save_status, panel.width - 150)
            self.screen.blit(
                self.font.render(status, True, (30, 25, 20)),
                (self.save_rect.right + 8, self.save_rect.y + 5),
            )

        if not actions:
            return
        self.selected %= len(actions)
        start = max(0, min(self.selected - 7, len(actions) - 15))
        for row, action in enumerate(actions[start : start + 15]):
            actual = start + row
            color = (70, 110, 75) if actual == self.selected else (30, 25, 20)
            label_text = fit_text(
                self.font,
                action_label(action, self.game),
                panel.width - 24,
            )
            label = self.font.render(label_text, True, color)
            position = (panel.x + 12, panel.y + 102 + row * 20)
            self.screen.blit(label, position)
            self.action_rects.append((pygame.Rect(position, (panel.width - 24, 21)), action))

    def choose_ai_action(self, legal_actions):
        player = self.acting_player
        state = public_game_state(self.observation_encoder, self.game, player).float()
        if self.game.ai_model is None:
            raise RuntimeError("The game has no shared AI model")
        with torch.no_grad():
            output = self.game.ai_model(state.unsqueeze(0))
            scores = (output.q_values if hasattr(output, "q_values") else output).squeeze(0)
        ranked = [(index, float(scores[index])) for index in legal_actions]
        return choose_ranked_ai_action(
            ranked,
            player.control,
            self.rng,
            dict(self.game.configuration.difficulty_top_k),
        )

    def save_current_game(self):
        try:
            filename = choose_save_file(self.game)
            if filename is None:
                return
            saved_path = save_game(
                self.game,
                filename,
                controller_rng_state=self.rng.getstate(),
            )
        except Exception as error:
            self.save_status = f"Save failed: {error}"
        else:
            self.save_status = f"Saved: {saved_path.name}"

    @property
    def acting_player(self):
        return self.game.players[self.game.active_player]

    def action_for_click(self, position, button, legal_actions):
        # Middle-click has no game meaning. In particular, it must not fall
        # through to the left-click Trader action while moving pieces.
        if button == 2:
            return None

        for rect, action in self.action_rects:
            if rect.collidepoint(position):
                return action
        layout = getattr(self, "layout", None)
        if layout is not None:
            for action, rect in layout.action_rects.items():
                if rect.collidepoint(position) and action in legal_actions:
                    return action

        if button == 1:
            upgrade_action = self._upgrade_action_for_click(position, legal_actions)
            if upgrade_action is not None:
                return upgrade_action

        post_index = 0
        for route_index, route in enumerate(self.game.selected_map.routes):
            for post in route.posts:
                if abs(position[0] - post.pos[0]) <= 24 and abs(position[1] - post.pos[1]) <= 24:
                    shape = PieceShape.MERCHANT if button == 3 else PieceShape.TRADER
                    action = DEFAULT_ACTION_CODEC.encode(PostInteraction(post_index, shape))
                    if action in legal_actions:
                        return action
                    replacement_action = DEFAULT_ACTION_CODEC.encode(
                        RouteInteraction(route_index, 0)
                    )
                    return replacement_action if replacement_action in legal_actions else None
                post_index += 1

        clicked_city = next(
            (
                city
                for city in self.game.selected_map.cities
                if city.x_pos <= position[0] <= city.x_pos + city.width
                and city.y_pos <= position[1] <= city.y_pos + city.height
            ),
            None,
        )
        if clicked_city is not None:
            route = self._route_toward_click(clicked_city, position)
            if route is not None:
                route_index = self.game.selected_map.routes.index(route)
                city_index = route.cities.index(clicked_city)
                if button == 1:
                    action = DEFAULT_ACTION_CODEC.encode(
                        RouteInteraction(route_index, ROUTE_OFFICE_SLOT_START + city_index)
                    )
                    return action if action in legal_actions else None
                if button == 3:
                    action = DEFAULT_ACTION_CODEC.encode(RouteInteraction(route_index, 0))
                    return action if action in legal_actions else None

        tile_rects = layout.tile_rects if layout is not None else {}
        for tile, rect in tile_rects.items():
            if rect.collidepoint(position):
                action = DEFAULT_ACTION_CODEC.encode(TileInteraction(TILE_SLOT_BY_TYPE[tile]))
                return action if action in legal_actions else None
        return None

    def _upgrade_action_for_click(self, position, legal_actions):
        selected_map = self.game.selected_map
        upgrades = [*selected_map.upgrade_cities, selected_map.specialprestigepoints]
        upgrade = next(
            (
                candidate
                for candidate in upgrades
                if pygame.Rect(
                    candidate.x_pos,
                    candidate.y_pos,
                    candidate.width,
                    candidate.height,
                ).collidepoint(position)
            ),
            None,
        )
        if upgrade is None:
            return None

        city = next(
            candidate for candidate in selected_map.cities if candidate.name == upgrade.city_name
        )
        candidates = []
        if upgrade.upgrade_type == "SpecialPrestigePoints":
            relative_x = max(0, min(upgrade.width - 1, position[0] - upgrade.x_pos))
            choice = int(relative_x * 4 / upgrade.width)
            for route in city.routes:
                route_index = selected_map.routes.index(route)
                action = DEFAULT_ACTION_CODEC.encode(
                    RouteInteraction(route_index, ROUTE_OUTCOME_SLOT_START + choice)
                )
                if action in legal_actions:
                    candidates.append((action, route))
        else:
            upgrade_index = city.upgrade_city_type.index(upgrade.upgrade_type)
            for route in city.routes:
                route_index = selected_map.routes.index(route)
                city_index = route.cities.index(city)
                action = DEFAULT_ACTION_CODEC.encode(
                    RouteInteraction(
                        route_index,
                        ROUTE_OUTCOME_SLOT_START + city_index * 2 + upgrade_index,
                    )
                )
                if action in legal_actions:
                    candidates.append((action, route))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        route_targets = []
        for action, route in candidates:
            other_city = next(endpoint for endpoint in route.cities if endpoint is not city)
            route_targets.append((action, route, other_city.midpoint))
        x_values = [target[2][0] for target in route_targets]
        y_values = [target[2][1] for target in route_targets]
        use_x_axis = max(x_values) - min(x_values) >= max(y_values) - min(y_values)
        axis = 0 if use_x_axis else 1
        ordered = sorted(route_targets, key=lambda target: target[2][axis])
        origin = upgrade.x_pos if use_x_axis else upgrade.y_pos
        extent = upgrade.width if use_x_axis else upgrade.height
        relative = max(0, min(extent - 1, position[axis] - origin))
        selection = int(relative * len(ordered) / extent)
        return ordered[selection][0]

    def _route_toward_click(self, city, position):
        controlled_routes = [
            route for route in city.routes if route.is_controlled_by(self.game.current_player)
        ]
        if not controlled_routes:
            return None
        if len(controlled_routes) == 1:
            return controlled_routes[0]

        center = (city.x_pos + city.width / 2, city.y_pos + city.height / 2)
        click_vector = (position[0] - center[0], position[1] - center[1])

        def alignment(route):
            other = route.cities[0] if route.cities[1] is city else route.cities[1]
            route_vector = (
                other.x_pos + other.width / 2 - center[0],
                other.y_pos + other.height / 2 - center[1],
            )
            length = max(1, (route_vector[0] ** 2 + route_vector[1] ** 2) ** 0.5)
            return (click_vector[0] * route_vector[0] + click_vector[1] * route_vector[1]) / length

        return max(controlled_routes, key=alignment)

    def run(self):
        running = True
        while running:
            actions = self.legal_actions()
            control = getattr(self.acting_player, "control", PlayerControl.HUMAN)
            self.screen.fill(TAN)
            self.layout = redraw_window(self.screen, self.game, actions)
            self.draw_action_browser(actions)
            if self.game.game_end:
                draw_end_game(self.screen, self.game.end_the_game())
            self.display.present()

            action_applied = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.display.resize(event.size)
                elif (
                    event.type == pygame.MOUSEBUTTONUP
                    and event.button == 1
                    and self.save_rect.collidepoint(self.display.to_logical(event.pos))
                ):
                    self.save_current_game()
                elif (
                    control.is_human
                    and not action_applied
                    and event.type == pygame.KEYDOWN
                    and actions
                    and not self.game.game_end
                ):
                    if event.key == pygame.K_UP:
                        self.selected = (self.selected - 1) % len(actions)
                    elif event.key == pygame.K_DOWN:
                        self.selected = (self.selected + 1) % len(actions)
                    elif event.key == pygame.K_e:
                        controls = [
                            DEFAULT_ACTION_CODEC.encode(ControlInteraction(slot)) for slot in (0, 1)
                        ]
                        control_action = next(
                            (action for action in controls if action in actions), None
                        )
                        if control_action is not None:
                            self.game.apply_action(control_action)
                            action_applied = True
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.game.apply_action(actions[self.selected % len(actions)])
                        action_applied = True
                elif (
                    control.is_human
                    and not action_applied
                    and event.type == pygame.MOUSEBUTTONUP
                    and actions
                    and not self.game.game_end
                ):
                    action = self.action_for_click(
                        self.display.to_logical(event.pos),
                        event.button,
                        actions,
                    )
                    if action is not None:
                        self.game.apply_action(action)
                        action_applied = True

            if running and not control.is_human and not self.game.game_end and actions:
                self.game.apply_ai_action(self.choose_ai_action(actions))
            self.clock.tick(30)
