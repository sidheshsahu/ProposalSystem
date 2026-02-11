from core.pipeline import UnifiedPipeline
from prompts.prompt_loader import load_prompt

def run_chat(document_store, history, query):
    prompt = load_prompt("chat.txt")
    pipeline = UnifiedPipeline(document_store, prompt)
    return pipeline.run(query, history)
