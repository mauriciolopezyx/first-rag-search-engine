import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .keyword_search_utils import (
    load_movies,
    BM25_K1,
    BM25_B
)


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, BM25_K1, BM25_B, limit)

    def weighted_search(self, query, alpha, limit):

        master_results = {}

        # both keyword and semantic results are list of dict's with: id, title, document, score

        bm25_results = self._bm25_search(query, limit*500)
        bm25_scores = [d["score"] for d in bm25_results]
        bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
        for bm25_res in bm25_results:
            id = bm25_res["id"]
            if not id in master_results:
                master_results[id] = {
                    "document": self.idx.docmap[id],
                    "keyword_score": 0,
                    "semantic_score": 0,
                    "hybrid_score": 0
                }
            master_results[id]["keyword_score"] = max(master_results[id]["keyword_score"], normalize(bm25_res["score"], bm25_min, bm25_max))

        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        semantic_scores = [d["score"] for d in semantic_results]
        semantic_min, semantic_max = min(semantic_scores), max(semantic_scores)
        for semantic_res in semantic_results:
            id = semantic_res["id"]
            if not id in master_results:
                master_results[id] = {
                    "document": self.idx.docmap[id],
                    "keyword_score": 0,
                    "semantic_score": 0,
                    "hybrid_score": 0
                }
            master_results[id]["semantic_score"] = max(master_results[id]["semantic_score"], normalize(semantic_res["score"], semantic_min, semantic_max))

        results = [{**d, "hybrid_score": hybrid_score(d["keyword_score"], d["semantic_score"], alpha)} for d in master_results.values()]
        
        sorted_results = sorted(results, key=lambda item:item["hybrid_score"], reverse=True)[:limit]
        return sorted_results


    # reciprocal rank fusion = rrf
    def rrf_search(self, query, k, limit):
        master_results = {}

        # both keyword and semantic results are list of dict's with: id, title, document, score

        bm25_results = self._bm25_search(query, limit*500)
        for i, bm25_res in enumerate(bm25_results):
            id = bm25_res["id"]
            if not id in master_results:
                master_results[id] = {
                    "document": self.idx.docmap[id],
                    "keyword_rank": float("inf"),
                    "semantic_rank": float("inf"),
                    "rrf_score": 0
                }
            master_results[id]["keyword_rank"] = min(master_results[id]["keyword_rank"], rrf_score(i+1, k))
            master_results[id]["rrf_score"] += master_results[id]["keyword_rank"]

        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        for i, semantic_res in enumerate(semantic_results):
            id = semantic_res["id"]
            if not id in master_results:
                master_results[id] = {
                    "document": self.idx.docmap[id],
                    "keyword_rank": float("inf"),
                    "semantic_rank": float("inf"),
                    "rrf_score": 0
                }
            master_results[id]["semantic_rank"] = min(master_results[id]["semantic_rank"], rrf_score(i+1, k))
            master_results[id]["rrf_score"] += master_results[id]["semantic_rank"]
        
        sorted_results = sorted(master_results.items(), key=lambda item:item[1]["rrf_score"], reverse=True)[:limit]
        return sorted_results

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize(val: float, val_min: float, val_max: float):
    return round((val - val_min) / (val_max - val_min), 4)

####

def weighted_search(query: str, alpha: float, limit: int):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.weighted_search(query, alpha, limit)

def ranked_search(query: str, k: int, limit: int):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.rrf_search(query, k, limit)