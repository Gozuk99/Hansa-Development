# Hansa Teutonica

A Python implementation of Hansa Teutonica intended to support deterministic
self-play and, eventually, reinforcement-learning agents.

The current priority is game-engine correctness. The project is not intended to
provide online multiplayer. Reinforcement-learning quality, position
evaluation, and suggested moves come after the rules and game lifecycle are
reliable.

## Current Capabilities

- Deterministic game creation from a map, player count, optional modules, and
  seed.
- Three-to-five-player games on the base, Eastern Hanseatic League, and
  Britannia maps. Two-player rules are intentionally out of scope.
- A Pygame New Game menu with every active seat defaulting to Human.
- Independently configurable Human, Easy, Medium, Hard, and
  Impossible ("Magnus") seats.
- Optional Mission Cards on Map 1 and optional Emperor's Favour tiles on every
  map.
- Standard bonus markers by default, with optional seeded or manually selected
  standard/promo replacement supplies.
- A legal-action-driven game window and a deterministic headless runner.
- Automated engine, configuration, rendering, and complete-game tests.

Exact save/load and replayable game-history support remain future milestones.

## Requirements

- Python 3.11
- Runtime dependencies from `requirements-ci.txt`
- Development and validation tools from `requirements-dev.txt`

Install dependencies in a virtual environment:

```console
python -m pip install -r requirements-ci.txt -r requirements-dev.txt
```

## Run the Interactive Game

```console
python hansa_game.py
```

The New Game menu configures the map, player count, controller for each seat,
Mission Cards, Emperor's Favour tiles, and replacement bonus-marker supply
before constructing the game. Press Enter to start or Escape to close the menu.

`sample_hansa_game.py` is retained only as a compatibility launcher and forwards
to the same entry point. Importing either launcher does not open a window, train
a model, or save a checkpoint.

### Game Controls

- Route post: left-click to place/displace with a Trader (square); right-click
  for a Merchant (circle).
- Controlled-route city: left-click to claim an office; right-click to complete
  the route without claiming an office (endpoint controllers still score).
- Drawn upgrade or Special Prestige box: left-click the desired upgrade or
  prestige value.
- Displacement: left-click to place a displaced Trader and right-click to place
  a displaced Merchant. After placing the displaced piece, use **Finish
  Displacement** when optional pieces are unavailable or declined. Additional
  pieces may use any shape available from the current stock/supply source that
  fits a nearest legal post.
- Legal-action browser: Up/Down selects an action and Enter applies it.
- Press `E` to finish the turn when End Turn is legal.

All GUI moves come from the engine's current legal-action mask and pass through
`Game.apply_action()`. When multiple routes or outcomes share a location, the
click position selects among them; every legal choice is also available in the
action browser.

## Run a Deterministic Headless Game

```console
python run_headless_game.py --map 2 --players 3 --seed 124
```

Optional setup modules are disabled unless explicitly requested:

```console
python run_headless_game.py --map 1 --players 3 --seed 124 \
  --mission-cards --emperors-favour
```

Mission Cards are valid only on Map 1. Emperor's Favour may be enabled on any
map. Promo markers are never included by default; provide an explicit
12-marker replacement supply by repeating `--bonus-marker TYPE` exactly twelve
times. The three fixed starting markers remain separate.

The same configuration and seed should produce the same action trace and final
scores. See `run_headless_game.py --help` for all available options.

## Validation

Run the same deterministic validation used by pull requests:

```console
python tools/validate_pr.py
```

It parses every Python file, runs Ruff lint and formatting checks, and executes
the complete `unittest` suite. Tests do not begin training or save model
checkpoints.

To run only the test suite:

```console
python -m unittest discover -s tests -v
```

Pull requests run the equivalent checks through
`.github/workflows/pull-request-validation.yml`. Independent Codex review is
advisory and runs outside GitHub Actions, so CI requires no OpenAI API key,
creates no OpenAI API billing, and never auto-merges.

## Architecture and Rules

- [Repository assessment](docs/REPOSITORY_ASSESSMENT.md)
- [First milestone plan](docs/FIRST_MILESTONE_PLAN.md)
- [New Game configuration](docs/NEW_GAME_CONFIGURATION.md)
- [Drawing architecture](docs/DRAWING_ARCHITECTURE.md)
- [Rules compliance matrix](docs/RULES_COMPLIANCE_MATRIX.md)
- [Big Box rulebook](docs/hansa-teutonica-big-box-rulebook.md)
- [Official FAQ](docs/Hansa_Teutonica_FAQ_v5.md)
- [Promo bonus-marker rules](docs/BigBoxPromoBonusMarkers.md)

`GameConfiguration` is the reusable boundary for validated setup. The game
engine owns state, legal-action masking, and mutation. The drawing layer renders
that state, maps user input to currently legal action indices, and does not own
rules legality.

## Development Priorities

1. Complete and verify game rules and turn structure.
2. Keep deterministic setup, action application, and complete-game tests green.
3. Implement exact state save/load and reproducible action history.
4. Add manual continuation, single-AI turns, and batch alternate-outcome
   simulation from saved positions.
5. Improve reinforcement-learning observations, rewards, models, evaluation,
   and suggested moves.
6. Improve statistics and presentation.

The long-term goal is a dependable engine that can generate and reproduce game
positions, complete games through self-play, and train AI opponents without
coupling game rules to the GUI or model implementation.
