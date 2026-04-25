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
    try:
        inverted_index = InvertedIndex()
        inverted_index.load()
        tokens = prepare_tokens(query)
        search_results = []
        for token in tokens:
            if len(search_results) >= 5:
                break
            for doc_id in sorted(inverted_index.get_documents(token)):
                if doc_id in inverted_index.docmap:
                    search_results.append({"id": doc_id, "title": inverted_index.docmap[doc_id]["title"]})
                    if len(search_results) >= 5:
                        break
    except FileNotFoundError:
        print("File not found!")
        return []
    return search_results

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

    def load(self):
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docmap_path) :
            raise FileNotFoundError
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)

