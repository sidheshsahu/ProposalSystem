"""Chat-based RAG service for interactive proposal queries.

This module enables natural language conversations about proposals,
allowing users to ask questions and receive context-aware answers based
on the proposal document content.
"""
from core.pipeline import UnifiedPipeline
from prompts.prompt_loader import load_prompt

def run_chat(document_store, history, query):
    prompt = load_prompt("chat.txt")
    pipeline = UnifiedPipeline(document_store, prompt)
    return pipeline.run_chat(query, history)
