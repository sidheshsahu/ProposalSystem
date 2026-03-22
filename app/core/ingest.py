"""PDF document ingestion and embedding pipeline.

This module handles the complete PDF ingestion workflow including:
- PDF parsing to extract text
- Chunking documents into manageable pieces
- Generating embeddings using sentence transformers
- Storing embeddings in the vector database
"""
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

def ingest_pdf(pdf_path, document_store):
    converter = PyPDFToDocument()
    splitter = DocumentSplitter(
        split_by="word",
        split_length=CHUNK_SIZE,
        split_overlap=CHUNK_OVERLAP
    )

    embedder = SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL
    )
    embedder.warm_up()

    docs = converter.run(sources=[pdf_path])["documents"]
    docs = splitter.run(docs)["documents"]
    docs = embedder.run(docs)["documents"]

    document_store.write_documents(docs)