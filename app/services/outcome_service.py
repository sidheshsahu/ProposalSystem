from core.pipeline import UnifiedPipeline
from prompts.prompt_loader import load_prompt

def run_outcome(document_store, notes):
    prompt = load_prompt("outcome.txt")
    pipeline = UnifiedPipeline(document_store, prompt)
    return pipeline.run("Estimate the overall acceptance chance of the proposal and provide recommendations to improve the chances of acceptance based on the organization context and retrieved context.", notes)
