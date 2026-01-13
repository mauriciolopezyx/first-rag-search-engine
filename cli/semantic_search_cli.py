#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_query,
    verify_embeddings,
    search
)
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies the semantic search model")
    subparsers.add_parser("verify_embeddings", help="Verifies the embeddings of movies.json")

    query_parser = subparsers.add_parser("embed_query", help="Embed query with the semantic search model")
    query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Semantically search for documents with a query")
    search_parser.add_argument("query", type=str, help="Query text")
    search_parser.add_argument("limit", type=float, nargs='?', default=5, help="Tunable results limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_query":
            embedding = embed_query(args.query)
            print(f"Query: {args.query}")
            print(f"First 5 dimensions: {embedding[:5]}")
            print(f"Shape: {embedding.shape}")
        case "verify_embeddings":
            embeddings = verify_embeddings()
            print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
        case "search":
            results = search(args.query, args.limit)
            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]} (score: {res["score"]})\n{res["description"][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()