#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch
from lib.search_utils import load_movies
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify the embedding model")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the input text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds the query")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

    search_parser = subparsers.add_parser("search", help="Searchs for semantically related movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5)

    chunk_parser = subparsers.add_parser("chunk", help="Searchs for semantically related movies")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "embed_query":
            embed_query_text(args.query)

        case "search":
            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(load_movies())
            for i, result in enumerate(semantic_search.search(args.query, args.limit), start=1):
                print(f"{i}. {result['title']} (score: {result['score']:.4f})\n{result['description'][:100]}")

        case "chunk":
            words = args.text.split()
            chunks = []
            for i in range(0, len(words), args.chunk_size):
                chunks.append(words[i:i + args.chunk_size])

            print(f"Chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {' '.join(chunk)}")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()