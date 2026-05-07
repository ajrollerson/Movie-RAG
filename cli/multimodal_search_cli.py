import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verifies image embedding")
    verify_parser.add_argument("image", help="Image for the embedding")

    image_search_parser = subparsers.add_parser("image_search", help="Searches using an image")
    image_search_parser.add_argument("image", help="Image for the search")

    args = parser.parse_args()

    if args.command == "verify_image_embedding":
        verify_image_embedding(args.image)

    if args.command == "image_search":
        results = image_search_command(args.image)
        for i, result in enumerate(results, start=1):
            print(f"{i}. {result["title"]} (similarity: {result["score"]:.3f})\n {result["description"]}")

    else:
        parser.print_help()
    
if __name__ == "__main__":
    main()