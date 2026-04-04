import json
import string
from .search_utils import load_movies, load_stopwords
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