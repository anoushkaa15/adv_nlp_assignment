from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


class OpenAIClientError(RuntimeError):
    """Raised when OpenAI cannot produce a usable response."""


@dataclass
class GrokClient:
    """
    OpenAI-based replacement while keeping the same class name
    so the rest of the project does not need changes.
    """

    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout: int = 60

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", self.model)

        if not self.api_key:
            raise OpenAIClientError(
                "Missing OpenAI API key. Set OPENAI_API_KEY in your .env file."
            )

        self.client = OpenAI(api_key=self.api_key)

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> dict[str, Any]:
        """
        Call OpenAI and parse JSON response.
        """

        text = self._complete(
            system_prompt,
            user_prompt,
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(text)

        except json.JSONDecodeError as exc:
            raise OpenAIClientError(
                f"Expected JSON response, received: {text[:500]}"
            ) from exc

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> str:
        """
        Call OpenAI and return plain text.
        """

        return self._complete(
            system_prompt,
            user_prompt,
            response_format=None
        )

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, str] | None
    ) -> str:

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.1,
                response_format=response_format
            )

            return response.choices[0].message.content

        except Exception as exc:
            raise OpenAIClientError(
                f"OpenAI API request failed: {exc}"
            ) from exc


class MockGrokClient:
    """Deterministic fallback for demos without API access."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        lowered_system = system_prompt.lower()

        if "credibility scorer" in lowered_system:
            return {
                "credibility_score": 12,
                "confidence_level": "medium",
                "misinformation_risk": "high",
                "justification": [
                    "The central claim is contradicted by retrieved fact-checking evidence.",
                    "Search results do not show a NASA announcement supporting the claim.",
                    "Confidence is medium because mock evidence is snippet-level rather than full article text.",
                ],
            }

        if "evidence analyst" in lowered_system:
            return {
                "supported_claims": [],
                "contradicted_claims": [
                    "NASA announced Earth will experience six days of total darkness next month because of a solar storm."
                ],
                "uncertain_claims": [
                    "Whether a specific viral post used altered NASA branding is not verifiable from snippets alone."
                ],
                "reasoning": [
                    "Retrieved fact-check snippets describe the six-days-of-darkness claim as a recurring hoax.",
                    "No evidence item reports an actual NASA announcement."
                ],
            }

        if "fact-checking claim extraction assistant" in lowered_system:

            if len(user_prompt.strip()) < 30 or "asdf" in user_prompt.lower():
                return {
                    "main_claim": "",
                    "sub_claims": [],
                    "entities": [],
                    "topics": ["unclear"],
                    "claim_type": "unclear",
                }

            return {
                "main_claim": "NASA announced Earth will experience six days of total darkness next month because of a solar storm.",
                "sub_claims": [
                    "NASA announced an upcoming six-day darkness event.",
                    "A solar storm will cause global darkness."
                ],
                "entities": [
                    "NASA",
                    "Earth",
                    "solar storm"
                ],
                "topics": [
                    "science",
                    "space",
                    "misinformation"
                ],
                "claim_type": "social media claim",
            }

        return {}

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        return """# Fake News Verification Report

## 1. Original Claim
Viral posts claim that NASA announced Earth will experience six days of total darkness next month because of a solar storm.

## 2. Extracted Factual Claims
- NASA allegedly announced a six-day global darkness event.
- A solar storm allegedly will cause this event next month.

## 3. Evidence Summary
The retrieved evidence does not support the alleged NASA announcement.

## 4. Supported vs. Contradicted Points
- Supported: None
- Contradicted: Main claim contradicted by evidence

## 5. Credibility Score
12 / 100

## 6. Confidence Level
Medium

## 7. Final Verdict
The claim is very likely false.

## 8. Limitations
Verification depends on available search snippets and retrieved sources.
"""