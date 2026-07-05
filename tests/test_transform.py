"""Unit tests for HA <-> Codex payload transformations."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from homeassistant.components.conversation import AssistantContent, ChatLog
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
import pytest

from custom_components.codex_conversation.transform import (
    async_prepare_files_for_prompt,
    build_input_items,
)


class _FakeHass:
    """Minimal HomeAssistant-like object for executor job calls."""

    async def async_add_executor_job(self, target, *args):
        return target(*args)


@pytest.mark.parametrize(
    ("filename", "mime_type", "expected_type"),
    [
        ("image.png", None, "input_image"),
        ("document.pdf", None, "input_file"),
    ],
)
async def test_async_prepare_files_for_prompt_supported_types(
    tmp_path: Path, filename: str, mime_type: str | None, expected_type: str
) -> None:
    file_path = tmp_path / filename
    file_path.write_bytes(b"test-bytes")

    result = await async_prepare_files_for_prompt(
        cast(HomeAssistant, _FakeHass()),
        [(file_path, mime_type)],
    )

    assert len(result) == 1
    assert result[0]["type"] == expected_type


async def test_async_prepare_files_for_prompt_rejects_unsupported_file_type(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello")

    with pytest.raises(HomeAssistantError, match="Only images and PDF"):
        await async_prepare_files_for_prompt(
            cast(HomeAssistant, _FakeHass()),
            [(file_path, None)],
        )


async def test_async_prepare_files_for_prompt_missing_file() -> None:
    with pytest.raises(HomeAssistantError, match="does not exist"):
        await async_prepare_files_for_prompt(
            cast(HomeAssistant, _FakeHass()),
            [(Path("/tmp/definitely-missing-file.png"), "image/png")],
        )


def test_build_input_items_preserves_assistant_text_tool_calls_and_native() -> None:
    tool_call = llm.ToolInput(
        id="call_1",
        tool_name="turn_on",
        tool_args={"entity_id": "light.kitchen"},
    )
    assistant_content = AssistantContent(
        agent_id="conversation.codex",
        content="Turning on the light.",
        thinking_content="Need to call the Home Assistant tool first.",
        tool_calls=[tool_call],
        native={"type": "reasoning", "id": "rs_1", "summary": []},
    )

    chat_log = cast(
        ChatLog,
        type("ChatLogStub", (), {"content": [assistant_content]})(),
    )

    items = build_input_items(chat_log)

    assert [item["type"] for item in items] == [
        "message",
        "function_call",
        "reasoning",
    ]
    assert items[0]["content"][0]["text"] == "Turning on the light."
    assert items[1]["name"] == "turn_on"
    assert items[2]["summary"] == [
        {
            "type": "summary_text",
            "text": "Need to call the Home Assistant tool first.",
        }
    ]
