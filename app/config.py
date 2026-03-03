import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_INDEX = "biaspredictor"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "arbiter"
