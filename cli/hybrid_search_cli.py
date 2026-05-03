import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import normalize, load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="Scores to normalize")
    
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Does a weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Query text")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)




    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize(args.scores)
            for score in results:
                print(f"* {score:.4f}")

        case "weighted-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result["title"]}\n  Hybrid Score: {result["score"]:.3f}\n  BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}\n  {result["description"][:100]}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()