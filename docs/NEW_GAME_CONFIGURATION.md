# New Game Configuration

`GameConfiguration` is the reusable boundary between the interface and game
initialization. The Pygame menu edits presentation-only `NewGameMenuState`, then
builds and validates one immutable configuration before any `Game` is created.

## Configuration Flow

1. `hansa_game.py` opens `NewGameMenu`.
2. The menu dynamically hides Mission Cards unless Map 1 is selected.
3. Every active player seat defaults to `Human`; each seat can independently
   select Easy, Medium, Hard, or Impossible ("Magnus").
4. Manual Emperor's Favour selection requires exactly one distinct tile per
   player. Random mode uses the seeded engine selection.
5. A custom bonus-marker supply remains off by default. Random mode creates a
   legal seeded standard/promo mix; manual mode selects all twelve exact marker
   copies from the complete standard-and-promo pool. The three fixed starting
   markers remain separate.
6. `GameConfiguration.create_game()` validates the complete selection, creates
   the engine, applies manual pools, and attaches controller metadata to each
   player.
7. `GameWindow` submits only indices from the engine's legal-action mask.
   Right-clicking a controlled route's city scores the route, left-clicking
   claims an office, and left-clicking a drawn upgrade box selects that upgrade.

Invalid configurations raise `ValueError` before game construction. This keeps
validation testable without opening a Pygame window and gives future maps or
optional modules one place to add their constraints.

## AI Difficulty

The default top-k thresholds are stored in `AI_DIFFICULTY_TOP_K`:

| Difficulty | Ranked legal moves considered |
| --- | ---: |
| Easy | 15 |
| Medium | 10 |
| Hard | 5 |
| Impossible ("Magnus") | 1 |

Non-Magnus AI choices use score-weighted random selection inside the configured
top-k set. Magnus always selects the highest-ranked legal action. Thresholds are
part of `GameConfiguration`, so tuning them does not require a menu or save
format change.
