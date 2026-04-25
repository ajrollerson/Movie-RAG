import argparse
from lib.keyword_search import search_command, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    subparsers.add_parser("build", help="Build the inverted index")


    args = parser.parse_args()



    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            for i, movie in enumerate(search_command(args.query), start=1):
                print(f"{i}. {movie['title']}")

        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()

        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            print(inverted_index.get_tf(args.doc_id, args.term))
            
        

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()