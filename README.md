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

### How to Run

Start the interactive game:

		python hansa_game.py

The New Game menu configures player count, map, every player's controller,
Mission Cards, Emperor's Favour tiles, and promotional bonus markers before
constructing the game. Every player defaults to Human. AI seats may independently
use Easy (top 15), Medium (top 10), Hard (top 5), or Impossible/Magnus (top 1);
these thresholds live in `game/game_config.py` rather than in the UI.

`sample_hansa_game.py` remains as a compatibility launcher and forwards to the
same entry point. Importing either module does not start a game, train a model,
or save checkpoints.

All setup choices are represented by one reusable `GameConfiguration`. Manual
Emperor's Favour selection requires exactly one distinct tile per player.
Custom bonus-marker supplies are disabled by default; random mode generates a
legal standard/promo mix, while manual mode selects all twelve exact copies from
the complete standard-and-promo pool. Mission Cards appear only for Map 1.

### Headless Engine Verification

Run the automated engine checks without loading neural-network models, opening a
pygame window, training, or saving model checkpoints:

		python -m unittest discover -s tests -v

### Pull Request Validation

Install the pinned runtime and development dependencies, then run the same
deterministic checks used for pull requests:

		python -m pip install -r requirements-ci.txt -r requirements-dev.txt
		python tools/validate_pr.py

The command parses every Python file, runs the configured Ruff correctness
checks, checks formatting for Python files changed on the current branch, and
runs the complete `unittest` suite. Static type checking is not yet enabled
because the project does not currently have a type-checker configuration.

Pull requests automatically run the equivalent checks through
`.github/workflows/pull-request-validation.yml`. Each pull request should link
its issue and request a fresh-context advisory review; see
`.github/CODEX_REVIEW.md`. AI review runs outside GitHub Actions: CI requires no
OpenAI API key, incurs no OpenAI API billing, and never approves or merges a
pull request automatically.

Run one deterministic headless game:

		python run_headless_game.py --map 2 --players 3 --seed 124

Optional setup modules are disabled by default. Enable them explicitly with:

		python run_headless_game.py --map 1 --players 3 --mission-cards --emperors-favour

Mission Cards are only valid on map 1. Emperor's Favour may be enabled on any map.

Promo bonus markers are never included by default. To use them, pass an explicit
12-marker replacement supply through `bonus_marker_supply` or repeat
`--bonus-marker TYPE` exactly 12 times with `run_headless_game.py`. The three
fixed starting markers remain separate, preserving 15 total markers.

The same map, player count, and seed should produce the same action trace and
final scores. The currently verified smoke configurations are map 2 with three
players (seeds 124 and 125) and map 1 with three players (seed 124).

Map 2 with four players and seed 124 is a known incomplete case: the current
baseline policy does not reach a terminal state within 10,000 actions. It is
not yet part of the supported verification matrix.

### How to Play:
**Left-Click** to claim or displace with square.
**Right-Click** to claim or displace with circle.

Admin Mode:
**Shift-Click** an Upgrade (yellow city), to auto upgrade an ability. Used for testing/training.

### Long Term Goals:
- Train Models for all 5 players - **Cannot do until game logic is complete.**
- ~~Mini Expansion Bonus Markers~~
- Evaluation Bar
- - Initially thinking to do just a breakdown of the score if the game ended immediately
- Computer generated suggestion for move.
- - When AI Models are build, would print top suggested move.
- Reward structure for more advanced scenarios
- - Ex: blocking East/West Connection completion
- Intro Screen to select players, maps, bonus markers.
- Generate a game state to evaluate.
