import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import normalize, normalize_results
from .constants import LIMIT, ALPHA

def hybrid_score(bm25_score, semantic_score, alpha=ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score

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
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=LIMIT):
        bm25_results = self._bm25_search(query, (limit * 500))
        semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
        

        
        combined_scores = {}
        for result in normalize_results(bm25_results):
            doc_id = result["id"]
            combined_scores[doc_id] = {
                "title": result["title"],
                "description": result["description"],
                "bm25_score": result["normalized_score"],
                "semantic_score": 0.0,  
            }


        for result in normalize_results(semantic_results):
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                "title": result["title"],
                "description": result["description"],
                "bm25_score": 0.0,
                "semantic_score": result["normalized_score"],  
            }
            elif doc_id in combined_scores:
                combined_scores[doc_id]["semantic_score"] = max(
                    combined_scores[doc_id]["semantic_score"],
                    result["normalized_score"],
                )

        hybrid_results = []
        for doc_id, data in combined_scores.items():
            data["score"] = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)
            data["id"] = doc_id 
            hybrid_results.append(data)

        return sorted(hybrid_results, key=lambda r: r["score"], reverse=True)[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")