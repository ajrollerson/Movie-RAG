import argparse
from lib.search_utils import normalize

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize(args.scores)
            for score in results:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()