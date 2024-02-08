### Project Goals:

1. To play Hansa Teutonica against an AI/Computer **ONLY** written using Python.
2. Build a NN and train using Reinforcement Learning with PyTorch
3. Become more familiar with AI, GitHub, and Pytorch

**NEVER** intending to allow a multiplayer/online functionality.

Inspired by the Chess.com features to view the position evaluation, and view suggested move(s).

### Priorities (in order):
1. Game Logic **(being worked on now)**
2. Clean code
3. Acquire a variety of game positions to be used for training.
	Ex1: 1 action away from ending the game.
	Ex2: 1 action away from running out of Bonus Markers
	Ex3: 1 action away from ending the game and losing.
6. Statistic tracking:
	Turns taken
	Upgrade Ability Prioritization
	Office/City Prioritization
5. Visual effects/beauty

### Prerequisites:
Python 3.11.4
import pygame
import sys
import random
import torch
import gc

### How to Run:
		python sample_hansa_game.py
You may change the map and players by editing the line in the sample_hansa_game.py:

		game = Game(map_num=2, num_players=5)

### How to Play:
**Left-Click** to claim or displace with square.
**Right-Click** to claim or displace with circle.

Admin Mode:
**Shift-Click** an Upgrade (yellow city), to auto upgrade an ability. Used for testing/training.

### Long Term Goals:
- Train Models for all 5 players - **Cannot do until game logic is complete.**
- Mini Expansion Bonus Markers
- Evaluation Bar
- - Initially thinking to do just a breakdown of the score if the game ended immediately
- Computer generated suggestion for move.
- - When AI Models are build, would print top suggested move.
- Reward structure for more advanced scenarios
- - Ex: blocking East/West Connection completion
- Intro Screen to select players, maps, bonus markers.
- Generate a game state to evaluate.
