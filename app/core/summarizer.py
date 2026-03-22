"""Document summarization utilities using LLMs.

This module provides functions to generate structured summaries of proposals
including acceptance and rejection reasons using the Groq LLM API.
"""
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
        You are an expert proposal analyst.

        Analyze the following proposal document and generate a response in STRICT JSON format.

        JSON structure:

        {
          "text": "Overall summary of the proposal in 5 to 10 lines",
          "accept": [
            "1. Reason",
            "2. Reason",
            "3. Reason",
            "4. Reason",
            "5. Reason"
          ],
          "reject": [
            "1. Reason",
            "2. Reason",
            "3. Reason",
            "4. Reason",
            "5. Reason"
          ]
        }

        Instructions:
        - The document is a **proposal**.
        - "text" must contain an overall summary of the proposal in **5 to 10 lines**.
        - "accept" must contain **exactly 5 numbered reasons** why the proposal should be accepted.
        - "reject" must contain **exactly 5 numbered reasons** why the proposal might be rejected.
        - Each reason must start with numbering like **1., 2., 3., 4., 5.**
        - Keep the points concise and meaningful.
        - Return **ONLY valid JSON**. Do not add any explanation or text outside the JSON.

        Proposal Document:
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



