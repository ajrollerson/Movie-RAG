from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import load_movies

def verify_image_embedding(image_path):
    movies = load_movies()
    multimodal_search = MultimodalSearch(movies)
    embedded_image = multimodal_search.embed_image(image_path)
    print(f"Embedding shape: {embedded_image.shape[0]} dimensions")

def image_search_command(image_path):
    movies = load_movies()
    multimodal_search = MultimodalSearch(movies)
    return multimodal_search.search_with_image(image_path)

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.documents = documents
        for document in documents:
            self.texts.append(f"{document['title']}: {document['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        image = Image.open(image_path)
        embedding = self.model.encode([image])
        return embedding[0]
    
    def search_with_image(self, image_path, limit=None):
        image_embedding = self.embed_image(image_path)
        results = []
        for i, text_embedding in enumerate(self.text_embeddings):
            score = cosine_similarity(image_embedding, text_embedding)
            truncated = int(score * 1000) / 1000
            doc = self.documents[i]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score": truncated,
            })  
        sorted_scores = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_scores[:5]
