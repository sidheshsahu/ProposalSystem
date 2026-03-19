import fitz
from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from config import LLM_MODEL

def llm_run(proposal_data):

        template = f"""
          A governance proposal has completed voting. Based on the following details, generate a concise 2–3 line summary of the outcome.

            Proposal Title:
            {proposal_data['title']}

            Proposal Summary:
            The summary contains three fields:
            - Text: overall description of the proposal
            - Accept: reasons for accepting
            - Reject: reasons for rejecting

            {proposal_data['summary']}

            Total Votes:
            {proposal_data['totalVotes']}

            Winning Choice:
            {proposal_data['winningChoice']}

            Winner Votes:
            {proposal_data['winnerVotes']}

            Instructions for AI:
            - First, briefly summarize the proposal in 1 short line.
            - Then, clearly state whether it was accepted or rejected based on the winning choice.
            - Calculate and include the percentage of votes received by the winning choice using:
            (winnerVotes / totalVotes) * 100
            - Keep the output strictly within 2–3 lines.
            - Do NOT use bullet points, Markdown, or extra formatting.
            - Do NOT include raw field names like "Text", "Accept", or "Reject".
            - Keep it clean, natural, and human-readable.
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


