"""Tests for Codex Responses API request serialization."""

from __future__ import annotations

import pytest

from custom_components.codex_conversation.codex_api.requests import CodexRequest


def _request(reasoning_summary: str) -> CodexRequest:
    """Build a minimal reasoning-model request."""
    return CodexRequest(
        model="gpt-5.6-luna",
        input=[],
        reasoning_summary=reasoning_summary,
    )


def test_reasoning_summary_off_is_omitted() -> None:
    """The UI's disabled value must not be sent to the API."""
    body = _request("off").to_body()

    assert body["reasoning"] == {"effort": "medium"}


@pytest.mark.parametrize(
    ("configured_summary", "api_summary"),
    [
        ("auto", "auto"),
        ("short", "concise"),
        ("concise", "concise"),
        ("detailed", "detailed"),
    ],
)
def test_reasoning_summary_is_normalized(
    configured_summary: str, api_summary: str
) -> None:
    """Configured summary values must match the API vocabulary."""
    body = _request(configured_summary).to_body()

    assert body["reasoning"]["summary"] == api_summary


def test_unknown_reasoning_summary_is_rejected() -> None:
    """Invalid stored values must fail before making an API request."""
    with pytest.raises(ValueError, match="Unsupported reasoning summary"):
        _request("invalid").to_body()
