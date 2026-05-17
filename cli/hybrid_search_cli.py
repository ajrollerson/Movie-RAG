import argparse
import time
from sentence_transformers import CrossEncoder
from lib.hybrid_search import HybridSearch
from lib.search_utils import normalize, load_movies
from lib.query_enhancement import llm_spell_check, llm_rewrite, llm_expand, llm_rerank, llm_batch, llm_evaluate

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes scores")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="Scores to normalize")
    
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Does a weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Query text")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Does a search using reciprocal ranks")
    rrf_search_parser.add_argument("query", type=str, help="Query text")
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument("--image", help="Optional search image")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", default=None, type=str, choices=["individual", "batch", "cross_encoder"], help="Rerank method")
    rrf_search_parser.add_argument("--evaluate", action="store_true", help="Provides LLM evaluation")


    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize(args.scores)
            for score in results:
                print(f"* {score:.4f}")

        case "weighted-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result["title"]}\n  Hybrid Score: {result["score"]:.3f}\n  BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}\n  {result["description"][:100]}...")

        case "rrf-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            query = args.query
            enhanced = None
            if args.enhance == "spell":
                enhanced = llm_spell_check(query)
            elif args.enhance == "rewrite":
                enhanced = llm_rewrite(query)
            elif args.enhance == "expand":
                enhanced = llm_expand(query)

            if enhanced is not None:
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            limit = args.limit
            if args.rerank_method == "individual" or args.rerank_method == "batch" or args.rerank_method == "cross_encoder":
                limit = args.limit * 5
            if args.image:
                results = hybrid_search.rrf_search(query, args.k, limit, args.image)
            else:
                results = hybrid_search.rrf_search(query, args.k, limit)
            if args.rerank_method == "individual":
                for result in results:
                    result["rerank_score"] = llm_rerank(query, result)
                    time.sleep(3)
                results.sort(key=lambda r: r["rerank_score"], reverse=True)
                top_results = results[:args.limit]
                print(f"Re-ranking top {args.limit} results using individual method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                for i, result in enumerate(top_results, start=1):
                    print(f"{i}. {result['title']}\n  Re-rank Score: {result['rerank_score']:.3f}/10\n  RRF Score: {result['rrf_score']:.3f}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}, Image Rank: {result['image_rank']}\n  {result['description'][:100]}...")
            elif args.rerank_method == "batch":
                ranked_ids = llm_batch(query, results)
                lookup = {result["id"]: result for result in results}
                ranked = [lookup[id] for id in ranked_ids if id in lookup]
                print(f"Re-ranking top {args.limit} results using batch method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                for i, result in enumerate(ranked[:args.limit], start=1):
                    print(f"{i}. {result['title']}\n  Re-rank Rank: {i}\n  RRF Score: {result['rrf_score']:.3f}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}, Image Rank: {result['image_rank']}\n  {result['description'][:100]}...")
            elif args.rerank_method == "cross_encoder":
                pairs = []
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                for result in results:
                    pairs.append([query, f"{result.get('title', '')} - {result.get('description', '')}"])
                scores = cross_encoder.predict(pairs)
                for i, score in enumerate(scores):
                    results[i]["cross_encoder_score"] = score
                results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
                print(f"Re-ranking top {args.limit} results using cross_encoder method...")
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                for i, result in enumerate(results[:args.limit], start=1):
                    print(f"{i}.  {result['title']}\n   Cross Encoder Score: {result['cross_encoder_score']:.3f}\n   RRF Score: {result['rrf_score']:.3f}\n   BM25 Rank: {result['bm25_rank']},  Semantic Rank: {result['semantic_rank']}, Image Rank: {result['image_rank']}\n   {result['description'][:100]}...")

            else:
                for i, result in enumerate(results[:args.limit], start=1):
                    print(f"{i}. {result['title']}\n  RRF Score: {result['rrf_score']:.3f}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}, Image Rank: {result['image_rank']}\n  {result['description'][:100]}...")
            if args.evaluate:
                evaluated_results = llm_evaluate(query, results)
                for i, score in enumerate(evaluated_results):
                    results[i]["rating"] = score
                for i, result in enumerate(results, start=1):
                    print(f"{i}. {result["title"]}: {result["rating"]}/3")


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()