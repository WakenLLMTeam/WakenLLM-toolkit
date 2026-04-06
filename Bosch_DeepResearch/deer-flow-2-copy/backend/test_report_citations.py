#!/usr/bin/env python3
"""Simple test to verify citations in report."""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "packages" / "harness"))

from deerflow.client import DeerFlowClient

def test_report_with_citations():
    """Generate a simple report and check for citations."""
    client = DeerFlowClient()

    prompt = "What are recent AI developments? Please provide a brief summary with sources."
    thread_id = "test-report-citations"

    print("Generating report with web search...")
    response = client.chat(prompt, thread_id=thread_id)

    print("\n=== Report Content ===")
    print(response[:500])
    print("\n...")

    # Check for citations
    has_citations = "[1]" in response or "[2]" in response
    has_references = "参考文献" in response or "References" in response

    print(f"\n=== Citation Check ===")
    print(f"Has numeric citations: {has_citations}")
    print(f"Has reference section: {has_references}")

    if has_citations and has_references:
        print("\n✓ Citations are working!")
        return True
    else:
        print("\n✗ Citations are missing")
        print(f"\nFull response:\n{response}")
        return False

if __name__ == "__main__":
    try:
        result = test_report_with_citations()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
