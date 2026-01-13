#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_text
)
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies the semantic search model")

    search_parser = subparsers.add_parser("embed_text", help="Embed text with the semantic search model")
    search_parser.add_argument("text", type=str, help="Text to embed")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embedding = embed_text(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()