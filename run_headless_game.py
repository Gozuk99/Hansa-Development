import argparse

from game.game_runner import GameRunError, run_game


def main():
    parser = argparse.ArgumentParser(description="Run one deterministic game without training.")
    parser.add_argument("--map", dest="map_num", type=int, default=2)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--seed", type=int, default=124)
    parser.add_argument("--max-actions", type=int, default=10_000)
    parser.add_argument("--mission-cards", action="store_true")
    parser.add_argument("--emperors-favour", action="store_true")
    parser.add_argument(
        "--bonus-marker",
        action="append",
        dest="bonus_marker_supply",
        help="Explicit supply marker type; repeat exactly 12 times to choose a promo mix.",
    )
    args = parser.parse_args()

    try:
        result = run_game(
            map_num=args.map_num,
            num_players=args.players,
            seed=args.seed,
            max_actions=args.max_actions,
            use_mission_cards=args.mission_cards,
            use_emperors_favour=args.emperors_favour,
            bonus_marker_supply=args.bonus_marker_supply,
        )
    except GameRunError as error:
        parser.exit(1, f"Headless game failed: {error}\n")

    print(f"Map: {result.map_num}")
    print(f"Players: {result.num_players}")
    print(f"Seed: {result.seed}")
    print(f"Actions: {result.action_count}")
    print(f"Terminal reason: {result.terminal_reason}")
    print(f"Final scores: {result.final_scores}")
    print("Invariants: passed")


if __name__ == "__main__":
    main()
