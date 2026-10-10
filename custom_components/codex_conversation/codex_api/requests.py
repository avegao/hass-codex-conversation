"""
Request model for the Codex Responses API.

Mirrors ``ResponsesApiRequest`` from codex-api/src/endpoint/responses.rs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_REASONING_SUMMARY_VALUES: dict[str, str | None] = {
    "auto": "auto",
    "short": "concise",
    "concise": "concise",
    "detailed": "detailed",
    "off": None,
}


def _serialize_reasoning_summary(value: str) -> str | None:
    """Translate a stored Home Assistant option to the API vocabulary."""
    try:
        return _REASONING_SUMMARY_VALUES[value]
    except KeyError as err:
        supported = ", ".join(sorted(_REASONING_SUMMARY_VALUES))
        raise ValueError(
            f"Unsupported reasoning summary {value!r}; expected one of: {supported}"
        ) from err


@dataclass
class CodexRequest:
    """
    Typed request for the Codex ``/responses`` endpoint.

    ``input`` is a list of message dicts in the OpenAI Responses API format::

        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}

    ``instructions`` maps to the top-level field of the same name (system prompt).

    Reasoning parameters (``reasoning``, ``text``, ``include``) are only
    included in the serialised body for models that support them
    (``gpt-5*`` / ``gpt-6*`` / ``o*`` series).
    """

    model: str
    input: list[dict[str, Any]]
    instructions: str = ""
    store: bool = False
    reasoning_effort: str = "medium"
    reasoning_summary: str = "auto"
    text_verbosity: str = "medium"
    tools: list[dict[str, Any]] = field(default_factory=list)

    def _is_reasoning_model(self) -> bool:
        return self.model.startswith(("gpt-5", "gpt-6", "o"))

    def to_body(self) -> dict[str, Any]:
        """Serialise to the JSON body expected by the Codex endpoint."""
        body: dict[str, Any] = {
            "model": self.model,
            "stream": True,
            "store": self.store,
            "input": self.input,
        }
        if self.instructions:
            body["instructions"] = self.instructions
        if self.tools:
            body["tools"] = self.tools
        if self._is_reasoning_model():
            reasoning = {"effort": self.reasoning_effort}
            reasoning_summary = _serialize_reasoning_summary(self.reasoning_summary)

            if reasoning_summary is not None:
                reasoning["summary"] = reasoning_summary

            body["reasoning"] = reasoning
            body["text"] = {"verbosity": self.text_verbosity}
            body["include"] = ["reasoning.encrypted_content"]
        return body
