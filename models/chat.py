from dotenv import load_dotenv
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret
from haystack import Pipeline

load_dotenv()

# -------------------------------
# Shared Components
# -------------------------------
document_store = PineconeDocumentStore(
    index="biaspredictor",
    metric="cosine",
    dimension=384,
    spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
)

query_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
query_embedder.warm_up()

retriever = PineconeEmbeddingRetriever(document_store=document_store)

llm = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    generation_kwargs={"temperature": 0.3},
)

# -------------------------------
# Chat Prompt
# -------------------------------
chat_template = """
You are a helpful assistant answering questions about a course proposal.
Answer ONLY using the provided documents.
If the answer is not found, say "I don't know".

Conversation so far:
{{ chat_history }}

Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

User question:
{{ query }}

Answer:
"""

prompt_builder = PromptBuilder(
    template=chat_template,
    required_variables=["query", "chat_history", "documents"]
)

# -------------------------------
# Pipeline
# -------------------------------
pipeline = Pipeline()
pipeline.add_component("text_embedder", query_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)

pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

# -------------------------------
# Chat Memory
# -------------------------------
chat_history = []

# -------------------------------
# Public Function (API calls this)
# -------------------------------
def run_rag_chat(query: str):
    global chat_history

    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    result = pipeline.run({
        "text_embedder": {"text": query},
        "prompt_builder": {
            "query": query,
            "chat_history": history_text
        }
    })

    answer = result["llm"]["replies"][0]

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer, chat_history
