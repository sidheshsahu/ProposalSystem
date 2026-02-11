from pathlib import Path

def load_prompt(name: str) -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
