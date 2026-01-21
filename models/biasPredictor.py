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
# Prompt
# -------------------------------
template = """
You are acting as a reviewer from the Computer Science (CSE) Department evaluating a course proposal.

Department Biases (CSE):
- Prefers strong technical foundations, coding assignments, and advanced software integration.
- Dislikes vague theoretical proposals with minimal implementation details or no programming aspects.

Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Task:
- Decide whether to VOTE "YES" or "NO"
- Give a short explanation (2–4 lines)


IMPORTANT OUTPUT RULES:
- No markdown formatting
- No **bold**, *, bullet points, or numbered lists
- Use plain text sentences only

Output format:
Department: Computer Science
Vote: [YES or NO]
Reason: [reason]

Question: {{ question }}
Answer:
"""

prompt_builder = PromptBuilder(
    template=template,
    required_variables=["documents", "question"]
)

# -------------------------------
# Pipeline
# -------------------------------
pipeline = Pipeline()
pipeline.add_component("query_embedder", query_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)

pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

# -------------------------------
# Public Function (API calls this)
# -------------------------------
def bias_summary(query: str) -> str:
    result = pipeline.run({
        "query_embedder": {"text": query},
        "prompt_builder": {"question": query}
    })
    return result["llm"]["replies"][0]
