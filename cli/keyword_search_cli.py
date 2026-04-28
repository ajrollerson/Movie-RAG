import argparse
import math
from lib.constants import BM25_K1, BM25_B
from lib.keyword_search import search_command, prepare_tokens, bm25_idf_command, bm25_tf_command, InvertedIndex

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

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

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
            
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()