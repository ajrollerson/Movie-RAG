import argparse
import math
from lib.keyword_search import search_command, prepare_tokens, InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    subparsers.add_parser("build", help="Build the inverted index")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Combines both term frequency and inverse document frequency metrics.")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

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

        case "idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            total_doc_count = len(inverted_index.docmap)
            search_term = prepare_tokens(args.term)[0]
            term_match_doc_count = len(inverted_index.index.get(search_term, []))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            total_doc_count = len(inverted_index.docmap)
            search_term = prepare_tokens(args.term)[0]
            term_match_doc_count = len(inverted_index.index.get(search_term, []))
            tf_value = inverted_index.get_tf(args.doc_id, args.term)
            idf_value = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = tf_value * idf_value
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()