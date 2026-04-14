"""
MockLLMClient: returns deterministic JSON for testing without an API key.
Scores are derived from the prompt text so tests are reproducible.
"""
import json
import re
from typing import Optional
from llm.base import LLMClientInterface


class MockLLMClient(LLMClientInterface):
    async def complete(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        # Extract PERCLOS value from prompt to make output contextual
        perclos = 0.0
        m = re.search(r"PERCLOS score:\s+([\d.]+)", user)
        if m:
            perclos = float(m.group(1))

        # Derive a plausible score
        image_score = min(1.0, perclos / 0.15 * 0.6)
        text_score  = 0.25
        audio_score = 0.10
        composite   = 0.30 * text_score + 0.50 * image_score + 0.20 * audio_score

        if composite >= 0.75:
            tier = "SEVERE"
        elif composite >= 0.55:
            tier = "MODERATE"
        elif composite >= 0.30:
            tier = "MILD"
        else:
            tier = "NONE"

        return json.dumps({
            "text_score":      round(text_score, 3),
            "image_score":     round(image_score, 3),
            "audio_score":     round(audio_score, 3),
            "composite_score": round(composite, 3),
            "severity_tier":   tier,
            "text_rationale":  "Moderate speed variance, minor steering corrections.",
            "image_rationale": f"PERCLOS={perclos:.3f} indicates eye closure pattern.",
            "audio_rationale": "No significant yawning detected.",
            "reasoning":       f"Composite={composite:.3f}; tier={tier}.",
        })
