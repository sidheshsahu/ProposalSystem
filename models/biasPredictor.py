from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from dotenv import load_dotenv
import os
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.builders import PromptBuilder
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersTextEmbedder,SentenceTransformersDocumentEmbedder

load_dotenv()

document_store = PineconeDocumentStore(
  index="biaspredictor",
  metric="cosine",
  dimension=384,
  spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
  )

all_docs = document_store.count_documents()
print(f"Total documents in Pinecone: {all_docs}")

pdf_converter = PyPDFToDocument()
splitter = DocumentSplitter(
    split_by="word",
    split_length=300,
    split_overlap=50
)

docs = pdf_converter.run(
    sources=[r"C:\Users\hp\Desktop\RAG\RAG\Blockchain_Course_Proposal.pdf"]
)["documents"]

split_docs = splitter.run(docs)["documents"]



doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

doc_embedder.warm_up()


embedded_docs = doc_embedder.run(split_docs)["documents"]


query_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

query_embedder.warm_up()

document_store.write_documents(embedded_docs)
print("Documents indexed into Pinecone")

retriever =PineconeEmbeddingRetriever(document_store=document_store)

# docs = retriever.run({"query_embedding": query_embedder.run({"text": query})["embedding"]})["documents"]
# print("Retrieved docs:", len(docs))

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

Output format:
Department: Computer Science
Vote: [YES or NO]
Reason: [reason]

Question: {{ question }}
Answer:
"""

prompt_builder =PromptBuilder(template=template,required_variables=["documents", "question"])

llm = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    generation_kwargs={"temperature": 0.3}
)



pipeline = Pipeline()
pipeline.add_component("query_embedder", query_embedder)
pipeline.add_component("retriever", retriever)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)
pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")



query = """
As a Computer Science department reviewer, analyze the proposal and vote YES or NO for its acceptance.
"""

result = pipeline.run({
    "query_embedder": {"text": query},
    "prompt_builder": {"question": query}
})

print("\n-------- CSE Department Voting Analysis --------\n")
print(result["llm"]["replies"][0])





