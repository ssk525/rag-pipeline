# Retrieval-Augmented Generation (RAG)

## Overview

Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge retrieved from a document corpus. This grounds the model's responses in factual data, significantly reducing hallucination.

RAG was introduced by Lewis et al. (2020) at Meta AI and has since become the standard approach for building knowledge-intensive AI applications.

## Why RAG Matters

### The Hallucination Problem
LLMs generate text based on patterns learned during training. When asked about specific facts, dates, or domain-specific information, they often produce plausible but incorrect answers — a phenomenon known as hallucination. Studies show that GPT-4 hallucinates in approximately 3-5% of responses on factual queries, while smaller models can hallucinate at rates of 15-25%.

### RAG as a Solution
By retrieving relevant documents before generation, RAG provides the LLM with a factual grounding:
1. **Reduced hallucination**: The model answers from retrieved evidence, not memory
2. **Up-to-date information**: No retraining needed when knowledge changes
3. **Verifiable answers**: Citations allow users to verify the source
4. **Domain specificity**: Works with any document corpus

## RAG Architecture

### Standard RAG Pipeline

The typical RAG pipeline consists of these stages:

1. **Document Ingestion**: Load documents (PDFs, web pages, databases)
2. **Chunking**: Split documents into manageable pieces (200-1000 tokens)
3. **Embedding**: Convert chunks to dense vector representations
4. **Indexing**: Store embeddings in a vector database
5. **Retrieval**: Find relevant chunks for a user query
6. **Generation**: Feed retrieved context + query to the LLM

### Chunking Strategies

Chunking is critical to RAG quality. Common approaches:

#### Fixed-Size Chunking
- Split text every N tokens with M token overlap
- Simple and predictable
- Can break mid-sentence or mid-paragraph
- Typical: 500 tokens with 50 token overlap

#### Semantic Chunking
- Split at natural boundaries (paragraphs, sections, topics)
- Preserves semantic coherence
- Variable chunk sizes
- Generally produces better retrieval quality

#### Comparison
Research shows that semantic chunking improves retrieval precision by 12-18% compared to fixed-size chunking on document QA tasks, though fixed-size chunking is more predictable for embedding models trained on fixed-length inputs.

## Advanced RAG Techniques

### Hybrid Search
Combining sparse (BM25/keyword) and dense (vector/semantic) retrieval:
- BM25 excels at exact keyword matching
- Vector search captures semantic similarity
- Reciprocal Rank Fusion (RRF) merges results from both

Hybrid search typically improves recall by 15-25% over vector-only search.

### Reranking
After initial retrieval, a reranker model re-scores candidates:
- Cross-encoder models (e.g., Cohere Rerank) attend to both query and document jointly
- Much more accurate than bi-encoder similarity but slower
- Typical pipeline: retrieve 20-50 candidates, rerank to top 5

Reranking improves precision@5 by 20-30% in production systems.

### Query Expansion
Using a small LLM call to expand or rephrase the query before retrieval:
- Generates multiple query variations
- Captures different phrasings of the same intent
- Can improve recall by 10-15%

## Evaluation Metrics

### Faithfulness
Measures whether the answer is supported by the retrieved context. A faithfulness score of 0.85 means 85% of claims in the answer are verifiable from the context.

### Answer Relevancy
Measures how well the answer addresses the original question. Uses embedding similarity between the question and the answer.

### Context Precision
Measures whether the retrieved contexts are relevant to answering the question. High precision means fewer irrelevant chunks in the context.

### Context Recall
Measures whether all information needed to answer the question was retrieved. Low recall means relevant information was missed.

### Ragas Framework
Ragas is the standard open-source framework for RAG evaluation. It computes all four metrics above using LLM-as-judge methodology and provides both aggregate and per-question scores.

## Production Considerations

### Latency Budget
- Embedding query: ~50ms
- Vector search: ~20-50ms
- BM25 search: ~10-30ms
- Reranking: ~100-300ms
- LLM generation: ~500-2000ms
- Total: typically 700-2500ms end-to-end

### Cost Management
- Embedding costs: $0.0001 per 1K tokens (OpenAI ada-002)
- LLM costs: $0.01-0.03 per 1K tokens (GPT-4)
- Vector DB: $0-50/month for small-medium corpora
- Reranking: $0.001 per search (Cohere)

### Metadata Filtering
Always filter before semantic search in production:
- By date range (recent documents only)
- By source type (only policies, only manuals)
- By access level (user permissions)

## Common Failure Modes

1. **Chunks too small** (<100 tokens): Lose context, retrieval becomes noisy
2. **Chunks too large** (>800 tokens): Dilute relevant information, waste context window
3. **No reranker**: Initial retrieval has low precision, garbage in = garbage out
4. **No evaluation**: "It works on my query" is not evaluation
5. **No citations**: Without source attribution, you've built a confident liar

## References

- Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023)
- Gao et al. "Retrieval-Augmented Generation for Large Language Models: A Survey" (2024)
