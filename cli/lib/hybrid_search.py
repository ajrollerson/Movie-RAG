import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import normalize, normalize_results
from .constants import LIMIT, ALPHA, K
from .multimodal_search import MultimodalSearch, verify_image_embedding, image_search_command

def hybrid_score(bm25_score, semantic_score, alpha=ALPHA):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=K):
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.multimodal_search = MultimodalSearch(documents)

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

    def rrf_search(self, query, k, limit=10, image_path=None):
        bm25_results = self._bm25_search(query, (limit * 500))
        semantic_results = self.semantic_search.search_chunks(query, (limit * 500))
        image_results = {}
        if image_path:
            image_results = self.multimodal_search.search_with_image(image_path, limit=limit * 500)

        combined_results = {}
        for i, result in enumerate(bm25_results):
            rank = i + 1
            id = result["id"]
            combined_results[id] = {
                "id": id,
                "title": result["title"],
                "description": result["description"],
                "bm25_rank": rank,
                "semantic_rank": 0,
                "image_rank": 0,
                "rrf_score": rrf_score(rank, k)
            }
        
        for i, result in enumerate(semantic_results):
            id = result["id"]
            rank = i + 1
            if id not in combined_results:
                combined_results[id] = {
                "id": id,
                "title": result["title"],
                "description": result["description"],
                "bm25_rank": 0,
                "semantic_rank": rank,
                "image_rank": 0,
                "rrf_score": rrf_score(rank, k)
            }
            else:
                combined_results[id]["semantic_rank"] = rank
                combined_results[id]["rrf_score"] += rrf_score(rank, k)
    
        if image_results:
            for i, result in enumerate(image_results):
                id = result["id"]
                rank = i + 1
                if id not in combined_results:
                    combined_results[id] = {
                    "id": id,
                    "title": result["title"],
                    "description": result["description"],
                    "bm25_rank": 0,
                    "semantic_rank": 0,
                    "image_rank": rank,
                    "rrf_score": rrf_score(rank, k)
                }
                else:
                    combined_results[id]["image_rank"] = rank
                    combined_results[id]["rrf_score"] += rrf_score(rank, k)

        
        return sorted(combined_results.values(), key=lambda x: x["rrf_score"], reverse=True)[:limit]
        