Readme – Movie RAG

RAG is a medium-scale retrieval project comprising BM25 search, semantic search, hybrid retrieval, reranking pipelines, and multimodal search functionality with LLM enhancement. Built as part of a guided project (boot.dev), extended to include an optional image parameter for Hybrid search.

Key skills and knowledge developed: 

•	Multi-stage retrieval system design. 
•	Embedding-based similarity systems. 
•	Ranking fusion strategies. 
•	CLI tool design.
•	Modular architecture thinking. 
•	Experimentation and evaluation mindset.

Key features:

The base project implemented the following retrieval functionality:

•	Hybrid BM25 + semantic retrieval. 
•	RRF fusion.
•	Reranking options (cross-encoder / LLM / batch). 
•	Query enhancement pipeline.

Personal additions made that enhance the base project:

•	Optional multimodal image retrieval integrated into the RRF pipeline.
•	Dynamic image ranking display within CLI output.
•	Experimental multimodal retrieval testing using CLIP embeddings.

Known limitations:

•	Limited multimodal grounding (image embeddings without captions). 
•	Reliance on a relatively clean and static dataset, which may not reflect noisy real-world retrieval environments.
•	Heuristic weighting in RRF rather than learned ranking.
•	Lack of large-scale evaluation or benchmark datasets for validating scoring metrics and retrieval performance.

Design choices:

•	Upon completing the guided project, I noticed the multimodal functionality operated largely as a separate retrieval pathway rather than participating directly in hybrid fusion. Thus, rather than developing a new pipeline I updated the existing hybrid search pipeline.
•	While I had considered updating the weighted search pipeline as well, it was unclear how the ‘hybrid_score’ function and weighting would need to be adjusted to accommodate a third parameter, without the appropriate experimental data.
Testing and observations:
•	During testing, image retrieval significantly altered ranking behaviour in ambiguous queries. For example, adding a Paddington image caused visually relevant films to outrank stronger textual matches, demonstrating the influence of multimodal fusion within the RRF pipeline.
•	A similar test was conducted with queries of varying ambiguity to retrieve Jurassic park, using dinosaur imagery. It was found with ambiguous queries, BM25 and Semantic Ranks varied wildly, while the use of image ranking reduced ambiguity, significantly altering RRF scores to bolster more relevant results.

Future improvements:

•	Adding caption generation or multimodal LLM grounding.
•	Learning-to-rank instead of heuristic fusion. 
•	Better dataset construction / evaluation framework.
•	UI layer or API wrapper.
•	Scaling indexing strategy.
