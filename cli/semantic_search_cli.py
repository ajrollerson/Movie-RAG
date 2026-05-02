#!/usr/bin/env python3

import argparse
import re
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_chunk, SemanticSearch, ChunkedSemanticSearch
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

    chunk_parser = subparsers.add_parser("chunk", help="Chunks input text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=1)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunks input sentences")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embeds chunks")

    search_chunked = subparsers.add_parser("search_chunked", help="Searches chunks")
    search_chunked.add_argument("query", type=str, help="Search query")
    search_chunked.add_argument("--limit", type=int, default=5)

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
            i = 0
            while i < len(words):               
                chunk = words[i:i + args.chunk_size]
                chunks.append(chunk)
                if i + args.chunk_size >= len(words):             
                    break
                i += args.chunk_size - args.overlap

            print(f"Chunking {len(args.text)} characters")
            for index, chunk in enumerate(chunks, start=1):
                print(f"{index}. {' '.join(chunk)}")

        case "semantic_chunk":
            sentence_chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for index, chunk in enumerate(sentence_chunks, start=1):
                print(f"{index}. {chunk}")

        case "embed_chunks":
            movies = load_movies()
            chunked_semantic_searcher = ChunkedSemanticSearch()
            embeddings = chunked_semantic_searcher.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            movies = load_movies()
            chunked_semantic_searcher = ChunkedSemanticSearch()
            embeddings = chunked_semantic_searcher.load_or_create_chunk_embeddings(movies)
            results = chunked_semantic_searcher.search_chunks(args.query, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"   {result["document"]}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()