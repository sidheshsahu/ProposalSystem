from core.pipeline import UnifiedPipelineChat
from prompts.prompt_loader import load_prompt

def run_chat(document_store, history, query):
    prompt = load_prompt("chat.txt")
    pipeline = UnifiedPipelineChat(document_store, prompt)
    return pipeline.run(query, history)
