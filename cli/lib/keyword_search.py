import json
import string
import os
import math
import pickle
from lib.constants import BM25_K1, BM25_B, LIMIT
from .search_utils import load_movies, load_stopwords, CACHE_DIR
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

clear_punctuation = str.maketrans("", "", string.punctuation)
stopwords = load_stopwords()
stemmer = PorterStemmer()


def prepare_tokens(text):
    tokens = text.translate(clear_punctuation).lower().split()
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

def bm25_idf_command(term):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_idf(term)

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    inverted_index = InvertedIndex()
    inverted_index.load()
    return inverted_index.get_bm25_tf(doc_id, term, k1=BM25_K1, b=BM25_B)

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = prepare_tokens(text)
        count = 0
        for token in tokens: 
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
            count += 1
        self.doc_lengths[doc_id] = count

    def __get_avg_doc_length(self) -> float:
        if self.doc_lengths == {}:
            return 0.0
        total = 0
        for length in self.doc_lengths.values():
            total += length
        return total / len(self.doc_lengths)

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
        tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        with open(tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        index_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        if not os.path.exists(index_path) or not os.path.exists(docmap_path) or not os.path.exists(tf_path) or not os.path.exists(self.doc_lengths_path):
            raise FileNotFoundError
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)
    
    def get_tf(self, doc_id, term):
        search_term = prepare_tokens(term)
        if len(search_term) != 1:
            raise Exception("More than one candidate term found!")
        return self.term_frequencies[doc_id][search_term[0]]
    
    def get_bm25_idf(self, term: str) -> float:
        tdc = len(self.docmap)
        tokens = prepare_tokens(term)
        if len(tokens) != 1:
            raise Exception("Term must be a single token!")
        df = len(self.index.get(tokens[0], []))
        idf = math.log((tdc - df + 0.5) / (df + 0.5) + 1)
        return idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm =  1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1) / (tf + k1 * length_norm))
    
    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit=LIMIT):
        query_tokens = prepare_tokens(query)
        scores = {}
        for doc_id in self.docmap:
            total_score = 0
            for token in query_tokens:
                total_score += self.bm25(doc_id, token)
            scores[doc_id] = total_score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_scores = sorted_scores[:limit]
        search_results = []
        for doc_id, score in top_scores:
            movie = self.docmap[doc_id]
            search_results.append({"doc_id": doc_id, "title": movie["title"], "score": score})
        return search_results