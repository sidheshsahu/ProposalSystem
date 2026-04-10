"""Prompt template loader for LLM interactions.

This module provides utilities to load and manage prompt templates stored
as text files, enabling flexible prompt management and version control.
"""
from pathlib import Path

def load_prompt(name: str) -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()




