from core.pipeline import UnifiedPipeline
from prompts.prompt_loader import load_prompt

def run_outcome(document_store, notes):
    prompt = load_prompt("outcome.txt")
    pipeline = UnifiedPipeline(document_store, prompt)
    return pipeline.run("Evaluate proposal", notes)
