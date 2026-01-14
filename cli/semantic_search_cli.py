#!/usr/bin/env python3

from lib.semantic_search import (
    verify_model,
    embed_query,
    verify_embeddings,
    search,
    chunk_query,
    chunk_semantic_query,
    embed_chunks,
    search_chunked
)
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies the semantic search model")
    
    subparsers.add_parser("verify_embeddings", help="Verifies the embeddings of movies.json")

    subparsers.add_parser("embed_chunks", help="Embeds the movies of movies.json on a chunk basis")


    query_parser = subparsers.add_parser("embed_query", help="Embed query with the semantic search model")
    query_parser.add_argument("query", type=str, help="Query text to embed")

    search_parser = subparsers.add_parser("search", help="Semantically search for documents with a query")
    search_parser.add_argument("query", type=str, help="Query text")
    search_parser.add_argument("limit", type=float, nargs='?', default=5, help="Tunable results limit")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Semantically search for documents that are chunked with a query")
    search_chunked_parser.add_argument("query", type=str, help="Query text")
    search_chunked_parser.add_argument("limit", type=float, nargs='?', default=5, help="Tunable results limit")

    chunk_parser = subparsers.add_parser("chunk", help="Split a query into of chunk_size words")
    chunk_parser.add_argument("query", type=str, help="Query text")
    chunk_parser.add_argument("chunk_size", type=int, nargs='?', default=200, help="Chunk word size")
    chunk_parser.add_argument("overlap", type=int, nargs='?', default=0, help="Overlap between chunks (in words)")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split a query into chunks of chunk_size sentences")
    semantic_chunk_parser.add_argument("query", type=str, help="Query text")
    semantic_chunk_parser.add_argument("chunk_size", type=int, nargs='?', default=4, help="Chunk sentence size")
    semantic_chunk_parser.add_argument("overlap", type=int, nargs='?', default=0, help="Overlap between chunks (in sentences)")

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
                print(f"{i+1}. {res["title"]} (score: {res["score"]})\n{res["document"]}...")
        case "chunk":
            print(f"Chunking {len(args.query)} characters")
            res = chunk_query(args.query, args.chunk_size, args.overlap)
            for i, s in enumerate(res):
                print(f"{i+1}. {s}")
        case "semantic_chunk":
            print(f"Chunking {len(args.query)} characters")
            res = chunk_semantic_query(args.query, args.chunk_size, args.overlap)
            for i, s in enumerate(res):
                print(f"{i+1}. {s}")
        case "embed_chunks":
            embeddings = embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            results = search_chunked(args.query, args.limit)
            for i, res in enumerate(results):
                print(f"\n{i+1}. {res["title"]} (score: {res["score"]:.4f})")
                print(f"   {res["document"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()