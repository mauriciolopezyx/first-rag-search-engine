from lib.hybrid_search import (
    normalize,
    weighted_search,
    ranked_search
)

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of floats between 0 and 1")
    normalize_parser.add_argument("values", nargs="+", type=float)

    weighted_search_parser = subparsers.add_parser("weighted_search", help="Find the best query results with a hybrid search of BM25 (keyword) and semantic search scores (NOT rankings)")
    weighted_search_parser.add_argument("query", type=str, help="Query text")
    weighted_search_parser.add_argument("alpha", type=float, nargs='?', default=0.5, help="Tunable parameter controlling weighting between keyword and semantic scoring (>0.5 = keyword biased, <0.5 = semantic biased)")
    weighted_search_parser.add_argument("limit", type=int, nargs='?', default=5, help="Search results limit")

    rrf_parser = subparsers.add_parser("ranked_search", help="Find the best query results with a hybrid search of BM25 (keyword) and semantic search rankings (NOT scores)")
    rrf_parser.add_argument("query", type=str, help="Query text")
    rrf_parser.add_argument("k", type=float, nargs='?', default=60, help="Tunable parameter controlling how much more weight we give to higher-ranked results (lower the k = more weight to higher-ranked)")
    rrf_parser.add_argument("limit", type=int, nargs='?', default=5, help="Search results limit")
    rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_parser.add_argument("--rerank-method", type=str, choices=["batch", "cross_encoder"], help="Results rerank method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            args_min, args_max = min(args.values), max(args.values)
            results = [normalize(val, args_min, args_max) for val in args.values]
            print("Normalized results are:", results)
        case "weighted_search":
            results = weighted_search(args.query, args.alpha, args.limit)

            # each result entry has fields document, keyword_score, semantic_score, and hybrid_score
            # where document is the FULL document from invertedIndex->docmap
            
            print("Hybrid Weighed Search Results\n-----")
            for i, res in enumerate(results):
                print(f"{i+1}. {res["document"]["title"]}\nHybrid Score: {res["hybrid_score"]}\nBM25: {res["keyword_score"]}, Semantic: {res["semantic_score"]}\nDescription: {res["document"]["description"][:100]}...\n")
        case "ranked_search":
            results = ranked_search(args.query, k=args.k, limit=args.limit, enhance=args.enhance, rerank_method=args.rerank_method)  
            if results:
                print("Hybrid Ranked Search Results\n-----")
                for i, res_obj in enumerate(results):
                    res = res_obj[1]
                    print(f"{i+1}. {res["document"]["title"]}\nRRF Score: {res["rrf_score"]}\nBM25 Rank: {res["keyword_rank"]}, Semantic Rank: {res["semantic_rank"]}\nDescription: {res["document"]["description"][:100]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()