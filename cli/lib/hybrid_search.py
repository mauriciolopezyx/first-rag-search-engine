import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder
from .keyword_search_utils import (
    load_movies,
    BM25_K1,
    BM25_B
)

from .search_utils import (
    spell_check_query,
    rewrite_query,
    expand_query,
    batch_rerank_results
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
            master_results[id]["keyword_rank"] = i+1
            master_results[id]["rrf_score"] += rrf_score(i+1, k)

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
            master_results[id]["semantic_rank"] =i+1
            master_results[id]["rrf_score"] += rrf_score(i+1, k)
        
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

def ranked_search(query: str, k: float=60, limit: int=5, enhance=None, rerank_method=None):
    enhanced_query = None
    initial_limit = limit

    if rerank_method:
        limit *= 5

    match enhance:
        case "spell":
            enhanced_query = spell_check_query(query)
        case "rewrite":
            enhanced_query = rewrite_query(query)
        case "expand":
            enhanced_query = expand_query(query)

    print("Original query:", query)
    if enhanced_query:
        print( f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
        query = enhanced_query

    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    initial_results = hybrid_search.rrf_search(query, k, limit)

    # results is list of tuple of (document id, actual result object) where weighted_search was already just a list of [actual result objects]
    
    if not rerank_method:
        return initial_results
    
    str_docs = []
    match rerank_method:
        case "batch":
            for res in initial_results:
                res_doc = res[1]["document"]
                str_docs.append(f"Id: {res_doc["id"]}\nTitle: {res_doc["title"]}\nDescription: {res_doc["description"]}\n")
    
            reranks = batch_rerank_results(query, str_docs)
            rerank_results = []
            for res in initial_results:
                rerank_idx = reranks.index(res[1]["document"]["id"]) if res[1]["document"]["id"] in reranks else float("inf")
                if rerank_idx == float("inf"):
                    print(res[1]["document"]["id"], "was not found (default to inf rank)")
                rerank_results.append({**res[1], "rerank_rank": rerank_idx+1})
            sorted_results = sorted(rerank_results, key=lambda item:item["rerank_rank"], reverse=False)[:initial_limit]
            print("Hybrid Ranked [Batch] Search Results\n-----")
            for i, res in enumerate(sorted_results):
                print(f"{i+1}. {res["document"]["title"]}\nRerank Rank: {res["rerank_rank"]}\nRRF Score: {res["rrf_score"]}\nBM25 Rank: {res["keyword_rank"]}, Semantic Rank: {res["semantic_rank"]}\nDescription: {res["document"]["description"][:100]}...\n")
        case "cross_encoder":
            # str_docs will be a list of pairs
            for res in initial_results:
                res_doc = res[1]["document"]
                str_docs.append([query, f"{res_doc["title"]} - {res_doc["description"]}"])
            scores = cross_encoder_results(str_docs)
            score_results = []
            for i, res in enumerate(initial_results):
                score_results.append({**res[1], "cross-encoder-score": scores[i]})
            sorted_results = sorted(score_results, key=lambda item:item["cross-encoder-score"], reverse=True)[:initial_limit]
            print("Hybrid Ranked [Cross Encoder] Search Results\n-----")
            for i, res in enumerate(sorted_results):
                print(f"{i+1}. {res["document"]["title"]}\nCross Encoder Score: {res["cross-encoder-score"]}\nRRF Score: {res["rrf_score"]}\nBM25 Rank: {res["keyword_rank"]}, Semantic Rank: {res["semantic_rank"]}\nDescription: {res["document"]["description"][:100]}...\n")


def cross_encoder_results(str_docs: list[list[str]]):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(str_docs)
    print("final scores:", scores)
    return scores
