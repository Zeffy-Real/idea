# RAG (Retrieval-Augmented Generation) Architecture

## Overview

Retrieval-Augmented Generation (RAG) is a technique that enhances large language model (LLM) outputs by grounding them in external knowledge sources. Instead of relying solely on the model's pre-trained knowledge, RAG retrieves relevant information from a knowledge base and incorporates it into the generation process.

## Core Components

### 1. Document Ingestion Pipeline

The ingestion pipeline is responsible for processing raw documents into a searchable format:

- **Document Loading**: Supports multiple formats including PDF, Markdown, TXT, and DOCX. Each loader extracts text and metadata (title, page count, word count) from the source file.

- **Text Chunking**: Documents are split into smaller, semantically coherent chunks. Common strategies include:
  - Fixed-size chunking with overlap
  - Sentence-aware splitting
  - Recursive text splitting (paragraphs → sentences → words)
  - Semantic chunking based on embedding similarity

- **Embedding**: Each chunk is converted into a dense vector representation using an embedding model. Popular models include BGE, sentence-transformers, and OpenAI embeddings.

- **Indexing**: Embeddings are stored in a vector database for efficient similarity search.

### 2. Retrieval Engine

The retrieval engine finds the most relevant chunks for a given query:

- **Semantic Search**: Uses cosine similarity between query and chunk embeddings
- **Keyword Search**: Traditional BM25 or TF-IDF matching for exact term matches
- **Hybrid Search**: Combines semantic and keyword search for better recall
- **Re-ranking**: Cross-encoder models re-order results by relevance
- **Query Expansion**: Techniques like HyDE (Hypothetical Document Embeddings) generate alternative queries to improve retrieval

### 3. Generation Layer

The generation layer produces grounded answers:

- **Prompt Engineering**: Constructs prompts that include retrieved context, conversation history, and citation instructions
- **Context Window Management**: Ensures retrieved context fits within the LLM's context window
- **Citation Grounding**: Every claim in the answer is traced back to a source document
- **Conversation Memory**: Maintains context across multi-turn conversations

## Production Considerations

### Error Handling

- Retry logic with exponential backoff for API calls
- Circuit breakers for dependent services
- Graceful degradation when components fail
- Comprehensive error classification (rate limits, timeouts, connection errors)

### Observability

- **Structured Logging**: JSON-formatted logs with request IDs, timestamps, and contextual metadata
- **Metrics**: Prometheus metrics for request latency, token usage, retrieval quality, and error rates
- **Tracing**: OpenTelemetry distributed tracing for end-to-end request visibility
- **Health Checks**: Liveness and readiness probes for container orchestration

### Security

- API key authentication
- Rate limiting per client
- Input validation and sanitization
- Sensitive data redaction in logs

### Scalability

- Stateless API design for horizontal scaling
- Connection pooling for database access
- Batch processing for embeddings
- Caching for frequently asked questions

## Vector Database Selection

Popular vector databases for RAG:

| Database | Strengths | Use Case |
|----------|-----------|----------|
| Qdrant | High performance, filtering, Rust-based | Production RAG |
| Pinecone | Managed service, auto-scaling | Serverless |
| Weaviate | GraphQL API, modules | Full-stack |
| Chroma | Simple, embedded | Prototyping |
| pgvector | PostgreSQL extension | Existing PG stacks |

## Embedding Model Selection

| Model | Dimension | Languages | Size |
|-------|-----------|-----------|------|
| BGE-m3 | 1024 | Multilingual | 2.2GB |
| BGE-small-zh-v1.5 | 512 | Chinese | 100MB |
| all-MiniLM-L6-v2 | 384 | English | 80MB |
| text-embedding-3-small | 1536 | Multilingual | API |

## Evaluation Metrics

- **Faithfulness**: Does the answer stick to the retrieved context?
- **Answer Relevance**: Is the answer relevant to the question?
- **Context Precision**: Are the retrieved chunks actually useful?
- **Context Recall**: Did we miss important information?
- **Citation Accuracy**: Are citations correctly mapped to sources?
