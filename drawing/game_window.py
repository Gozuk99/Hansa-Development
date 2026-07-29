"""Minimal legal-action-driven Pygame window for an initialized game."""

from __future__ import annotations

import random

import pygame
import torch

from ai.action_options import masking_out_invalid_actions
from ai.game_state import BoardData
from drawing.action_ui import action_label, phase_prompt
from drawing.drawing_utils import draw_end_game, redraw_window
from drawing.scaled_display import ScaledDisplay
from game.game_config import PlayerControl, choose_ranked_ai_action
from map_data.constants import MAX_ROUTES, TAN


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
        self.board_data = BoardData()
        self.rng = random.Random(game.seed)
        self.action_rects: list[tuple[pygame.Rect, int]] = []
        self.layout = None

    def legal_actions(self) -> list[int]:
        return masking_out_invalid_actions(self.game).nonzero(as_tuple=True)[0].tolist()

    def draw_action_browser(self, actions):
        self.action_rects.clear()
        panel = pygame.Rect(self.game.selected_map.map_width + 810, 10, 280, 440)
        pygame.draw.rect(self.screen, (245, 238, 218), panel)
        pygame.draw.rect(self.screen, (45, 38, 30), panel, 2)
        control = getattr(self.game.current_player, "control", PlayerControl.HUMAN)
        heading = self.font.render(
            f"Player {self.game.current_player.order}: {control.value}",
            True,
            (30, 25, 20),
        )
        self.screen.blit(heading, (panel.x + 10, panel.y + 10))
        help_text = self.font.render("Up/Down + Enter; E ends when legal", True, (30, 25, 20))
        self.screen.blit(help_text, (panel.x + 10, panel.y + 34))
        prompt_text = phase_prompt(self.game)
        while self.font.size(prompt_text)[0] > panel.width - 20 and " " in prompt_text:
            prompt_text = prompt_text.rsplit(" ", 1)[0] + "…"
        prompt = self.font.render(prompt_text, True, (117, 70, 42))
        self.screen.blit(prompt, (panel.x + 10, panel.y + 54))
        city_help = self.font.render(
            "City — L: office  M: upgrade  R: points",
            True,
            (30, 25, 20),
        )
        self.screen.blit(city_help, (panel.x + 10, panel.y + 74))

        if not actions:
            return
        self.selected %= len(actions)
        start = max(0, min(self.selected - 7, len(actions) - 15))
        for row, action in enumerate(actions[start : start + 15]):
            actual = start + row
            color = (70, 110, 75) if actual == self.selected else (30, 25, 20)
            label = self.font.render(action_label(action, self.game), True, color)
            position = (panel.x + 12, panel.y + 102 + row * 20)
            self.screen.blit(label, position)
            self.action_rects.append((pygame.Rect(position, (panel.width - 24, 21)), action))

    def choose_ai_action(self, legal_actions):
        player = self.game.current_player
        state = self.board_data.get_game_state(self.game).float()
        with torch.no_grad():
            scores = player.hansa_nn(state.unsqueeze(0)).squeeze(0)
        ranked = [(index, float(scores[index])) for index in legal_actions]
        return choose_ranked_ai_action(
            ranked,
            player.control,
            self.rng,
            dict(self.game.configuration.difficulty_top_k),
        )

    def action_for_click(self, position, button, legal_actions):
        for rect, action in self.action_rects:
            if rect.collidepoint(position):
                return action
        layout = getattr(self, "layout", None)
        if layout is not None:
            for action, rect in layout.action_rects.items():
                if rect.collidepoint(position) and action in legal_actions:
                    return action

        post_index = 0
        for route_index, route in enumerate(self.game.selected_map.routes):
            for post in route.posts:
                if abs(position[0] - post.pos[0]) <= 24 and abs(position[1] - post.pos[1]) <= 24:
                    action = post_index + (121 if button == 3 else 0)
                    if action in legal_actions:
                        return action
                    replacement_action = 543 + route_index
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
                    action = 242 + MAX_ROUTES + route_index * 2 + city_index
                    return action if action in legal_actions else None
                if button == 3:
                    action = 242 + route_index
                    return action if action in legal_actions else None
                if button == 2:
                    base = 242 + MAX_ROUTES * 3 + route_index * 4
                    choices = [
                        action for action in range(base, base + 4) if action in legal_actions
                    ]
                    city_choices = [
                        action for action in choices if ((action - base) // 2) == city_index
                    ]
                    special_city = next(
                        (
                            city
                            for city in route.cities
                            if "SpecialPrestigePoints" in city.upgrade_city_type
                        ),
                        None,
                    )
                    available = choices if special_city is not None else (city_choices or choices)
                    if not available:
                        return None
                    relative_x = max(
                        0, min(clicked_city.width - 1, position[0] - clicked_city.x_pos)
                    )
                    selection = int(relative_x * len(available) / clicked_city.width)
                    return available[selection]

        tile_actions = {
            "DisplaceAnywhere": 535,
            "+1Action": 536,
            "+1IncomeIfOthersIncome": 537,
            "+1DisplacedPiece": 538,
            "+4PtsPerOwnedCity": 539,
            "+7PtsPerCompletedAbility": 540,
        }
        tile_rects = layout.tile_rects if layout is not None else {}
        for tile, rect in tile_rects.items():
            if rect.collidepoint(position):
                action = tile_actions[tile]
                return action if action in legal_actions else None
        return None

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
            control = getattr(self.game.current_player, "control", PlayerControl.HUMAN)
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
                    elif event.key == pygame.K_e and 618 in actions:
                        self.game.apply_action(618)
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
                self.game.apply_action(self.choose_ai_action(actions))
            self.clock.tick(30)
