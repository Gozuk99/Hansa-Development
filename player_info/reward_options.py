# reward_attributes.py

import sys
from map_data.constants import COLOR_NAMES

WIN_REWARD = 2000
NEUTRAL = 0


class Rewards:
    def __init__(self, selection):
        self.end_game_on_your_turn = NEUTRAL

        # upgrades
        self.upgraded_keys = NEUTRAL
        self.upgraded_privilige = NEUTRAL
        self.upgraded_circles = NEUTRAL
        self.upgraded_actions = NEUTRAL
        self.upgraded_bank = NEUTRAL
        self.upgraded_bonus_points = NEUTRAL
        self.upgrade_finished_privilige = NEUTRAL
        self.upgrade_finished_circles = NEUTRAL
        self.upgrade_finished_actions = NEUTRAL
        self.upgrade_finished_bank = NEUTRAL

        # city
        self.city_new_owner = NEUTRAL
        self.city_claim_office = NEUTRAL
        self.city_claim_office_with_points = NEUTRAL
        # self.claim_office_increasing_max_route = NEUTRAL

        # route
        self.route_completed_east_west = NEUTRAL
        self.route_complete_got_points = NEUTRAL
        self.route_receive_bm = NEUTRAL

        # post
        self.post_adjacent_to_upgrade_city = NEUTRAL
        self.post_with_bm = NEUTRAL
        self.post_with_perm_bm = NEUTRAL
        self.post_with_nothing = NEUTRAL

        # displace/block
        self.displace_block_full_route = NEUTRAL
        # self.block_east_west
        # self.displace_to_block

        # bonus_marker
        self.bm_4actions = NEUTRAL
        self.bm_3actions = NEUTRAL
        self.bm_upgrade_anything = NEUTRAL
        self.bm_move3 = NEUTRAL
        self.bm_swap_city_office = NEUTRAL
        self.bm_place_adjacent = NEUTRAL

        self.assign_reward_structure(selection)

    def assign_reward_structure(self, selection):
        self.brown_priv_count = NEUTRAL
        self.blue_priv_count = NEUTRAL

        if selection == 1:
            self.assign_rs1()
        elif selection == 2:
            self.assign_rs2()
        elif selection == 3:
            self.assign_rs3()

    def assign_rs1(self):
        # assuming each point (of 20) is equal to 100
        self.end_game_on_your_turn = 150

        # upgrades
        self.upgraded_keys = 120
        self.upgraded_privilege = 150
        self.upgraded_circles = 150
        self.upgraded_actions = 180
        self.upgraded_bank = 130
        self.upgraded_bonus_points = 500
        self.upgrade_finished_privilige = 400
        self.upgrade_finished_circles = 500
        self.upgrade_finished_actions = 500
        self.upgrade_finished_bank = 400

        # city
        self.city_new_owner = 30
        self.city_claim_office = 30
        self.city_claim_office_with_points = 100
        # self.claim_office_increasing_max_route = NEUTRAL

        # route
        self.route_completed_east_west = 700
        self.route_complete_got_points = 100
        self.route_complete_receive_bm = 150
        self.route_complete_perm_bm = 70

        # post
        self.post_adjacent_to_upgrade_city = 5
        self.post_with_bm = 4
        self.post_with_perm_bm = 4
        self.post_with_nothing = 0.1

        # displace/block
        self.displace_block_full_route = NEUTRAL
        # self.block_east_west
        # self.displace_to_block

        # bonus_marker
        self.bm_4actions = 200
        self.bm_3actions = 150
        self.bm_upgrade_anything = 200
        self.bm_move3 = 100
        self.bm_swap_city_office = 100
        self.bm_place_adjacent = 50

    def get_end_game_placement_RL_rewards(self, game):
        # Fixed rewards by rank
        rank_rewards = {
            1: 2000,  # Reward for 1st place
            2: 500,  # Reward for 2nd place
            3: 0,  # Reward for 3rd place
            4: -500,  # Reward for 4th place
            5: -2000,  # Reward for 5th place
        }

        # Sort players based on their scores (higher scores are better)
        sorted_players = sorted(game.players, key=lambda x: x.final_score, reverse=True)
        num_players = len(sorted_players)

        # Initialize rewards for each rank
        player_rewards = {}
        last_score = None
        last_rank = 0
        shared_rank = 1

        # Assign rewards based on ranking and handle ties
        for i, player in enumerate(sorted_players):
            if player.final_score == last_score:
                player_rewards[player] = player_rewards[
                    last_player
                ]  # Same reward as previous for a tie
            else:
                if i > 0:  # Adjust rank only if not the first player
                    shared_rank = i + 1
                player_rewards[player] = rank_rewards.get(
                    shared_rank, -2000
                )  # Default to -2000 if rank beyond 5

            # Update for next iteration
            last_score = player.final_score
            last_player = player
            last_rank = shared_rank

        # Apply calculated rewards to players
        for player, reward in player_rewards.items():
            player.reward += reward
            print(
                f"{COLOR_NAMES[player.color]} (Rank {last_rank}/{num_players}): Score = {player.final_score}, Reward = {player.reward}"
            )

        # This handles the case where the active player ends the game
        game.players[game.active_player].reward += self.end_game_on_your_turn
