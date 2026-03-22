"""Document store initialization using Pinecone vector database.

This module provides utilities for initializing and configuring the Pinecone
vector store that handles document embeddings and similarity search operations.
"""
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from config import PINECONE_INDEX

def get_document_store(namespace:str):
    return PineconeDocumentStore(
        index=PINECONE_INDEX,
        dimension=384,
        namespace=namespace,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
