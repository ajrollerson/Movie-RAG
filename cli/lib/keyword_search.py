import json
import string
import os
import pickle
from .search_utils import load_movies, load_stopwords, CACHE_DIR
from collections import defaultdict
from nltk.stem import PorterStemmer

clear_punctuation = str.maketrans("", "", string.punctuation)
stopwords = load_stopwords()
stemmer = PorterStemmer()


def prepare_tokens(text):
    tokens = text.translate(clear_punctuation).lower().split(" ")
    filtered_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            filtered_tokens.append(stemmer.stem(token))
    return filtered_tokens

def search_command(query):
    movies = load_movies()
    results = []
    for movie in movies:
        clean_movie_title = prepare_tokens(movie["title"])
        clean_query = prepare_tokens(query)
        for query_token in clean_query:
            for title_token in clean_movie_title:
                if query_token in title_token:
                    results.append(movie)
                    break
    return results[:5]

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}

    def __add_document(self, doc_id, text):
        tokens = prepare_tokens(text)
        for token in tokens: 
            self.index[token].add(doc_id)

    def get_documents(self, term):
        doc_ids = self.index.get(term.lower(), set())
        return sorted(doc_ids)

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]] = movie
            movie_text = f"{movie['title']} {movie['description']}"
            self.__add_document(movie["id"], movie_text)

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

