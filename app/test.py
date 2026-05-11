from core.pipeline import UnifiedPipeline
from prompts.prompt_loader import load_prompt
from core.document_store import get_document_store


def run_bias(document_store, bias_text):
    prompt = load_prompt("bias.txt")
    pipeline = UnifiedPipeline(document_store, prompt)
    return pipeline.run("Evaluate whether the proposal should pass or not based on the bias criteria and the provided context.", bias_text)

result = run_bias(
            document_store=get_document_store("sahu"),
            bias_text=f"""
            The proposal is to implement a new feature that allows users to customize their profiles with themes and backgrounds.
            """
        )

print(result)