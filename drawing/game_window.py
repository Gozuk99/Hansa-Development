"""Minimal legal-action-driven Pygame window for an initialized game."""

from __future__ import annotations

import random

import pygame
import torch

from ai.action_options import masking_out_invalid_actions
from ai.game_state import BoardData
from drawing.drawing_utils import draw_end_game, redraw_window
from drawing.scaled_display import ScaledDisplay
from game.game_config import PlayerControl, choose_ranked_ai_action
from map_data.constants import TAN


def action_label(index: int) -> str:
    if index < 121:
        return f"Post {index}: Trader"
    if index < 242:
        return f"Post {index - 121}: Merchant"
    if index < 522:
        return f"Route/office/upgrade action {index}"
    if index < 527:
        return f"Income choice {index - 522}"
    if index == 618:
        return "Finish / End turn"
    return f"Context action {index}"


class GameWindow:
    def __init__(self, game):
        self.game = game
        self.width = game.selected_map.map_width + 800
        self.height = game.selected_map.map_height
        self.display = ScaledDisplay((self.width, self.height), "Hansa Teutonica")
        self.screen = self.display.canvas
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.selected = 0
        self.board_data = BoardData()
        self.rng = random.Random(game.seed)
        self.action_rects: list[tuple[pygame.Rect, int]] = []

    def legal_actions(self) -> list[int]:
        return masking_out_invalid_actions(self.game).nonzero(as_tuple=True)[0].tolist()

    def draw_action_browser(self, actions):
        self.action_rects.clear()
        panel = pygame.Rect(self.width - 290, 10, 280, 420)
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

        if not actions:
            return
        self.selected %= len(actions)
        start = max(0, min(self.selected - 7, len(actions) - 15))
        for row, action in enumerate(actions[start : start + 15]):
            actual = start + row
            color = (70, 110, 75) if actual == self.selected else (30, 25, 20)
            label = self.font.render(action_label(action), True, color)
            position = (panel.x + 12, panel.y + 65 + row * 22)
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

        post_index = 0
        for route in self.game.selected_map.routes:
            for post in route.posts:
                if abs(position[0] - post.pos[0]) <= 24 and abs(position[1] - post.pos[1]) <= 24:
                    action = post_index + (121 if button == 3 else 0)
                    return action if action in legal_actions else None
                post_index += 1

        board = self.game.current_player.board
        for index, rect in enumerate(getattr(board, "circle_buttons", ())):
            if rect.collidepoint(position):
                action = 522 + index
                return action if action in legal_actions else None

        end_turn = pygame.Rect(
            self.game.selected_map.map_width + 415,
            self.game.selected_map.map_height - 170,
            75,
            70,
        )
        if end_turn.collidepoint(position) and 618 in legal_actions:
            return 618

        tile_actions = {
            "DisplaceAnywhere": 535,
            "+1Action": 536,
            "+1IncomeIfOthersIncome": 537,
            "+1DisplacedPiece": 538,
            "+4PtsPerOwnedCity": 539,
            "+7PtsPerCompletedAbility": 540,
        }
        for tile, rect in getattr(self.game, "tile_rects", {}).items():
            if rect.collidepoint(position):
                action = tile_actions[tile]
                return action if action in legal_actions else None
        return None

    def run(self):
        running = True
        while running:
            actions = self.legal_actions()
            control = getattr(self.game.current_player, "control", PlayerControl.HUMAN)
            self.screen.fill(TAN)
            redraw_window(self.screen, self.game)
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
