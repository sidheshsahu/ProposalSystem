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


chat_history = []


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

chat_prompt_builder = PromptBuilder(template=chat_template,required_variables=["query","chat_history","documents"])

llm = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    generation_kwargs={"temperature": 0.3}
)

chat_pipeline = Pipeline()

chat_pipeline.add_component("text_embedder", query_embedder)
chat_pipeline.add_component("retriever", retriever)
chat_pipeline.add_component("prompt_builder", chat_prompt_builder)
chat_pipeline.add_component("llm", llm)
chat_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
chat_pipeline.connect("retriever.documents", "prompt_builder.documents")
chat_pipeline.connect("prompt_builder", "llm")

def rag_chat(query: str):
    global chat_history

    history_text = ""
    for msg in chat_history[:]:
        history_text += f"{msg['role']}: {msg['content']}\n"

    result = chat_pipeline.run({
        "text_embedder": {"text": query},
        "prompt_builder": {
            "query": query,
            "chat_history": history_text
        }
    })

    answer = result["llm"]["replies"][0]

    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    return answer


print(rag_chat("What is summary of the proposal?"))
print(chat_history)

