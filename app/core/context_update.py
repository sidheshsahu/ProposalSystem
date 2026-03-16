import fitz
from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from config import LLM_MODEL

def llm_run(proposal_data):

        template = f"""
           A governance proposal has completed voting. Consider the following details for generating a short, reusable organizational context:
.

            Proposal Title:
            {proposal_data['title']}

            Proposal Summary:
            In summary there are three fields Texts,Accept and Reject.
            Texts means summary of proposal.
            Accept means reason for accepting proposal.
            Reject means reasons for rejecting proposal

            {proposal_data['summary']}

            Total Votes:
            It gives us an idea of how many members participated in the voting process, indicating the level of engagement and interest in the proposal.

            {proposal_data['totalVotes']}

            Winning Details:
            It indicates the outcome of the proposal, showing which choice received the most votes and was ultimately accepted or rejected. This information is crucial for understanding the direction the organization has decided to take based on member input.

            {proposal_data['winningChoice']}

            
            
            Instructions for AI:
            - Generate a concise 2-3 line **organizational context**.
            - The context should guide future decision-making, capturing the rationale, impact, and implications for the organization.
            - **Do NOT include any specific proposal details** such as title, subject, numbers, or exact reasons.
            - **Do NOT mention any topic-specific content** (e.g., blockchain, students, etc.).
            - **Do NOT add bullet points, lists, or Markdown**.
            - Make it **generic** and reusable for all future proposals.
            -Do NOT include any prefix like "Here's a concise 2-3 line organizational context:".
            - Do NOT add bullet points, Markdown, or extra explanation.
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

 
        result = pipe.run({})

        output = result["llm"]["replies"][0]
        clean_output = output.replace("*", "")
        return clean_output


