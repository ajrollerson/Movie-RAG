import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit


    with open('data/golden_dataset.json', 'r') as file:
        golden_dataset = json.load(file)

    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    print(f"k={limit}")
    print("")
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]  
        results = hybrid_search.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [r["title"] for r in results]
        count = len(set(retrieved_titles) & set(relevant_docs))
        precision = count / len(results)
        recall = count / len(test_case["relevant_docs"])
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
    
    

if __name__ == "__main__":
    main()