#!/usr/bin/env python3
"""Test citation integration with DeerFlow agent."""

import asyncio
import json
from pathlib import Path
from deerflow.client import DeerFlowClient

async def test_citations():
    """Test that citations appear in agent output."""
    client = DeerFlowClient()

    # Simple prompt that triggers web search
    prompt = "What are the latest developments in AI in 2026?"
    thread_id = "test-citation-thread"

    print("Testing citation integration...")
    print(f"Prompt: {prompt}\n")

    # Stream the response
    response_text = ""
    async for event in client.stream(prompt, thread_id=thread_id):
        if event.get("type") == "values":
            messages = event.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    content = last_msg.content
                    if isinstance(content, str):
                        response_text = content
                        print(f"Response preview: {content[:200]}...\n")

    # Check for citations
    has_numeric_citations = "[1]" in response_text or "[2]" in response_text
    has_reference_section = "参考文献" in response_text or "References" in response_text

    print(f"\n=== Citation Check ===")
    print(f"Has numeric citations [1][2]...: {has_numeric_citations}")
    print(f"Has reference section: {has_reference_section}")
    print(f"\nFull response:\n{response_text}")

    return has_numeric_citations and has_reference_section

if __name__ == "__main__":
    result = asyncio.run(test_citations())
    exit(0 if result else 1)
