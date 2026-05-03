import json
import os

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORD_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt" )
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

def load_movies():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORD_PATH, "r") as f:
        return f.read().splitlines()
    
def normalize(scores):
    if not scores:
        return []
    normalized = []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    for score in scores:
        normalized.append((score - min_score) / (max_score - min_score))
    return normalized

def normalize_results(results):
    scores = [r["score"] for r in results]
    normalized = normalize(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]
    return results