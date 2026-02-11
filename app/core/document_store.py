from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from config import PINECONE_INDEX

def get_document_store():
    return PineconeDocumentStore(
        index=PINECONE_INDEX,
        dimension=384,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
