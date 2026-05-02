from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import json
from .search_utils import load_movies, SCORE_PRECISION
from .constants import LIMIT


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0) -> list[str]:
    txt = text.strip()
    if not txt:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", txt)
    if len(sentences) == 1 and not txt.endswith((".","!","?")):
        sentences = [txt]
    sentence_chunks = []
    i = 0
    while i < len(sentences):               
        chunk = sentences[i : i + max_chunk_size]
        if sentence_chunks and len(chunk) <= overlap:
            break

        cleaned_sentences = []
        for c in chunk:
            stripped = c.strip()
            if stripped:
                cleaned_sentences.append(c.strip())
    
        if cleaned_sentences:
            sentence_chunks.append(" ".join(cleaned_sentences))
        i += max_chunk_size - overlap
    return sentence_chunks

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("Input text is empty!")
        embedding = self.model.encode([text])
        return embedding[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            movie_strings.append(f"{document['title']}: {document['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document["id"]] = document
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
            return self.build_embeddings(documents)
        return self.build_embeddings(documents)
        
    def search(self, query, limit=LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        scores = []
        for i, document in enumerate(self.embeddings):
            scores.append((cosine_similarity(query_embedding, self.embeddings[i]), self.documents[i]))
        sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
        sliced_sorted_scores = sorted_scores[:limit] 
        results = []
        for score, doc in sliced_sorted_scores:
            results.append({"score": score, "title": doc["title"], "description": doc["description"]})
        return results

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {} 
        
        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc 
            description = doc["description"]   
            if not description:
                continue
            chunks = semantic_chunk(description, max_chunk_size=4, overlap=1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks),
                })
            
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for document in self.documents:
            self.document_map[document["id"]] = document
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        chunk_score = []
        for index, embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, embedding)
            chunk_score.append({
                "chunk_idx": self.chunk_metadata[index]["chunk_idx"],
                "movie_idx": self.chunk_metadata[index]["movie_idx"],
                "score": score
            })
        movie_scores = {}
        for chunk in chunk_score:
            if chunk["movie_idx"] not in movie_scores or chunk["score"] > movie_scores[chunk["movie_idx"]]["score"]:
                movie_scores[chunk["movie_idx"]] = chunk
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        results = sorted_movie_scores[:limit]
        formatted_results = []
        for movie_idx, chunk in results:
            doc = self.documents[movie_idx]
            formatted_results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc["description"][:100],        
                "score": round(chunk["score"], SCORE_PRECISION),           
                "metadata": doc.get("metadata") or {}       
            })
        return formatted_results
