"""Document ingestion pipeline: loading, chunking, embedding, and indexing."""
from cognita.ingestion.pipeline import IngestionPipeline
from cognita.ingestion.loaders import DocumentLoader
from cognita.ingestion.chunkers import TextChunker

__all__ = ["IngestionPipeline", "DocumentLoader", "TextChunker"]
