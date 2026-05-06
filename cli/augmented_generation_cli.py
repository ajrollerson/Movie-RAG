import argparse
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from lib.query_enhancement import llm_ag, llm_ags

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("-k", type=int, default=60)
    rag_parser.add_argument("--limit", type=int, default=5)

    summarise_parser = subparsers.add_parser("summarize", help="Perform RAG (search + generate a summary)")
    summarise_parser.add_argument("query", type=str, help="Search query for RAG")
    summarise_parser.add_argument("-k", type=int, default=60)
    summarise_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.rrf_search(query, args.k, args.limit)
            response = llm_ag(query, results)
            print("Search Results:")
            for result in results:
                print(f"- {result["title"]}")
            print("RAG Response:")
            print(f"{response}")

        case "summarize":
            query = args.query
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.rrf_search(query, args.k, args.limit)
            response = llm_ags(query, results)
            print("Search Results:")
            for result in results:
                print(f"- {result["title"]}")
            print("LLM Summary:")
            print(f"{response}")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()