from core.pipeline import UnifiedPipeline
from prompts.prompt_loader import load_prompt

def run_bias(document_store, bias_text):
    prompt = load_prompt("bias.txt")
    pipeline = UnifiedPipeline(document_store, prompt)
    return pipeline.run("Evaluate whether the proposal should pass or not based on the bias criteria and the provided context.", bias_text)