import fitz
from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from config import LLM_MODEL

def generic_summarizer(pdf_path):
        
        doc = fitz.open(pdf_path)

        text = ""
        for page in doc:
          text += page.get_text()

        documents = [Document(content=text)]
      
        template = """
        Summarize the following document.

        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}
        """

        prompt_builder = PromptBuilder(template=template)

        generator = OpenAIGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model=LLM_MODEL,
        generation_kwargs={"temperature": 0.25}
        )

        pipe = Pipeline()

        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", generator)

        pipe.connect("prompt_builder", "llm")

 
        result = pipe.run({
        "prompt_builder": {"documents": documents}
        })

        return result["llm"]["replies"][0]