from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.utils import Secret
from config import EMBEDDING_MODEL, LLM_MODEL
from dotenv import load_dotenv
import os
load_dotenv()

class UnifiedPipeline:
    def __init__(self, document_store, prompt_template):
        self.embedder = SentenceTransformersTextEmbedder(
            model=EMBEDDING_MODEL
        )
        self.embedder.warm_up()

        self.retriever = PineconeEmbeddingRetriever(
            document_store=document_store
        )

        self.prompt = PromptBuilder(
            template=prompt_template,
            required_variables=["documents"]
        )

        self.llm = OpenAIGenerator(
                api_key=Secret.from_env_var("GROQ_API_KEY"),
                api_base_url="https://api.groq.com/openai/v1",
                model=LLM_MODEL,
                generation_kwargs={"temperature": 0.25}
            )

        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", self.embedder)
        self.pipeline.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt", self.prompt)
        self.pipeline.add_component("llm", self.llm)

        self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.pipeline.connect("retriever.documents", "prompt.documents")
        self.pipeline.connect("prompt", "llm")

    def run(self, query, extra=""):
        result = self.pipeline.run({
            "embedder": {"text": query+" "+extra},
        })
        return result["llm"]["replies"][0]
    
    def run_chat(self, query, extra=""):
        result = self.pipeline.run({
            "embedder": {"text": query},
            "prompt": {"extra": extra}
        })
        return result["llm"]["replies"][0]