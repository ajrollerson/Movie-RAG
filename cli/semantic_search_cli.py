#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify", help="Verify the embedding model")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify the embeddings")
    

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed the input text")
    embed_text_parser.add_argument("text", type=str, help="Text to be embedded")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embeds the query")
    embed_query_parser.add_argument("query", type=str, help="Query to be embedded")

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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()