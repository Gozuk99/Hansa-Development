# Drawing Architecture

The Pygame layer presents the engine; it does not decide legality or mutate game
state. Every submitted move must be present in the engine's current legal-action
mask and is applied through `Game.apply_action()`.

## Responsibilities

- `drawing/new_game_menu.py` edits presentation-only setup state and creates one
  validated `GameConfiguration`.
- `drawing/scaled_display.py` owns the resizable physical window, fixed logical
  canvas, letterboxing, and physical-to-logical pointer translation.
- `drawing/action_ui.py` translates indexed actions and `TurnPhase` values into
  human-readable choices and prompts. It owns the presentation meaning of reused
  contextual indices.
- `drawing/drawing_utils.py` renders maps, writing desks, markers, tiles, scores,
  and phase-specific buttons. A render pass returns `DrawLayout` hitboxes and
  does not store them on `Game`, `Player`, `Board`, or marker objects.
- `drawing/game_window.py` owns the frame loop, legal-action browser, mouse and
  keyboard mapping, AI turn scheduling, and the single call to
  `Game.apply_action()`.

## Rules-Sensitive Presentation

- Mission cards appear only on the active player's writing desk.
- The replacement bonus-marker supply displays its remaining count, not its
  shuffled face-down contents.
- Opponents' used bonus markers display only face-down counts. Their types become
  selectable through the legal-action list only while resolving the promotional
  Exchange Bonus Marker.
- End Turn is clickable only when action `618` is legal.
- Indices `522–526` describe ordinary Income, Tribute income, or mandatory
  two-piece selection according to the current `TurnPhase`.
- Indices `535–542` describe tile purchases, tile payments, or the optional
  Income Favour response according to pending engine state.
- Replacement routes, office claims, upgrades, Special Prestige values,
  Additional Trading Posts, and bonus-marker workflows have descriptive labels.

## Interaction Conventions

- Route posts: left-click Trader; right-click Merchant.
- Controlled-route city: left-click office; right-click route points.
- Drawn upgrade box: left-click the desired ability or Special Prestige value.
- When several controlled routes meet at one city, clicking toward a route
  selects that route.
- When an upgrade graphic contains several choices, such as Actions/Bank in
  Waren or the four Special Prestige values, each drawn choice is its own
  left-click target. Every choice also remains available in the legal-action
  browser.
- Arrow keys navigate the legal-action browser; Enter applies the selected
  action; `E` finishes when action `618` is legal.

The numbered action encoding remains an engine/RL compatibility boundary. New UI
controls should use `action_label()`, the legal mask, and returned `DrawLayout`
hitboxes instead of duplicating action legality.
