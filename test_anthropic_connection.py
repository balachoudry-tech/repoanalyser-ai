
"""Verify that CodePilot AI can communicate with Anthropic."""

from src.llm.answer_generator import AnthropicAnswerGenerator


def main():
    """Run a small real-provider smoke test."""
    generator = AnthropicAnswerGenerator()

    context_result = {
        "context": """
[S1]
File: src/auth/service.py
Symbol: authenticate_user
Lines: 10-15

```python
def authenticate_user(username, password):
    return verify_password(username, password)
```
""",
        "sources": [
            {
                "source_id": "S1",
                "file_path": "src/auth/service.py",
                "symbol_name": "authenticate_user",
                "start_line": 10,
                "end_line": 15,
                "language": "python",
            }
        ],
    }

    result = generator.generate_answer(
        question="Where is user authentication implemented?",
        context_result=context_result,
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f'{source["source_id"]}: '
            f'{source["file_path"]}:'
            f'{source["start_line"]}-{source["end_line"]}'
        )

    print("\nUSAGE:")
    print(result["usage"])


if __name__ == "__main__":
    main()
