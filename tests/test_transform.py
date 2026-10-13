"""Unit tests for HA <-> Codex payload transformations."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from homeassistant.components.conversation import AssistantContent, ChatLog
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
import pytest

from custom_components.codex_conversation.transform import (
    async_prepare_files_for_prompt,
    build_input_items,
)

from .conftest import make_chat_log


@pytest.mark.parametrize("error", [False, True])
def test_build_input_items_modern_tool_result(
    monkeypatch: pytest.MonkeyPatch, error: bool
) -> None:
    """Modern results preserve their envelope without reading the deprecated alias."""
    import custom_components.codex_conversation.transform as transform

    data = {
        "today": date(2026, 10, 13),
        "timestamp": datetime(2026, 10, 13, 15, 0),
        "message": "tool failed" if error else "tool succeeded",
    }

    class ModernToolResultContent:
        tool_call_id = "call_modern"
        result = SimpleNamespace(data=data, error=error)

        @property
        def tool_result(self):
            raise AssertionError("The deprecated tool_result property was read")

    monkeypatch.setattr(transform, "ToolResultContent", ModernToolResultContent)

    items = build_input_items(make_chat_log([ModernToolResultContent()]))

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call_modern",
            "output": json.dumps(
                {
                    "data": {
                        "today": "2026-10-13",
                        "timestamp": "2026-10-13T15:00:00",
                        "message": data["message"],
                    },
                    "error": error,
                }
            ),
        }
    ]


def test_build_input_items_legacy_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older Core versions keep the original bare payload and date encoding."""
    import custom_components.codex_conversation.transform as transform

    class LegacyToolResultContent:
        tool_call_id = "call_legacy"
        tool_result = {"today": date(2026, 3, 3), "error": "legacy error"}

    monkeypatch.setattr(transform, "ToolResultContent", LegacyToolResultContent)

    items = build_input_items(make_chat_log([LegacyToolResultContent()]))

    assert items == [
        {
            "type": "function_call_output",
            "call_id": "call_legacy",
            "output": '{"today": "2026-03-03", "error": "legacy error"}',
        }
    ]


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


def test_format_tool_uses_probatio_when_voluptuous_openapi_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The converter must follow Home Assistant's 2026.9 dependency change."""
    import voluptuous as vol

    import custom_components.codex_conversation.transform as transform

    calls: list[tuple[object, object]] = []

    def fake_to_openapi(
        schema: object, *, custom_serializer: object = None
    ) -> dict[str, object]:
        calls.append((schema, custom_serializer))
        return {"type": "object"}

    probatio = ModuleType("probatio")
    setattr(probatio, "to_openapi", fake_to_openapi)
    monkeypatch.setitem(sys.modules, "probatio", probatio)
    monkeypatch.setitem(sys.modules, "voluptuous_openapi", None)

    def serializer(schema: object) -> dict[str, str]:
        return {"type": "string"}

    tool = MagicMock(spec=llm.Tool)
    tool.name = "ping"
    tool.description = None
    tool.parameters = vol.Schema({})

    result = transform.format_tool(tool, custom_serializer=serializer)

    assert result["parameters"] == {"type": "object"}
    assert calls == [(tool.parameters, serializer)]


def test_format_tool_falls_back_when_probatio_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy converter remains available on older Home Assistant versions."""
    import voluptuous as vol

    import custom_components.codex_conversation.transform as transform

    calls: list[tuple[object, object]] = []

    def fake_convert(
        schema: object, *, custom_serializer: object = None
    ) -> dict[str, object]:
        calls.append((schema, custom_serializer))
        return {"type": "object"}

    legacy_converter = ModuleType("voluptuous_openapi")
    setattr(legacy_converter, "convert", fake_convert)
    monkeypatch.setitem(sys.modules, "voluptuous_openapi", legacy_converter)

    def missing_probatio(name: str) -> ModuleType:
        raise ModuleNotFoundError("No module named 'probatio'", name="probatio")

    monkeypatch.setattr(transform, "import_module", missing_probatio)

    def serializer(schema: object) -> dict[str, str]:
        return {"type": "string"}

    tool = MagicMock(spec=llm.Tool)
    tool.name = "ping"
    tool.description = None
    tool.parameters = vol.Schema({})

    result = transform.format_tool(tool, custom_serializer=serializer)

    assert result["parameters"] == {"type": "object"}
    assert calls == [(tool.parameters, serializer)]


def test_format_tool_falls_back_when_probatio_has_no_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy converter handles a probatio module without the new API."""
    import voluptuous as vol

    import custom_components.codex_conversation.transform as transform

    calls: list[tuple[object, object]] = []
    probatio = ModuleType("probatio")

    def load_probatio(name: str) -> ModuleType:
        return probatio

    def fake_convert(
        schema: object, *, custom_serializer: object = None
    ) -> dict[str, object]:
        calls.append((schema, custom_serializer))
        return {"type": "object"}

    legacy_converter = ModuleType("voluptuous_openapi")
    setattr(legacy_converter, "convert", fake_convert)
    monkeypatch.setitem(sys.modules, "voluptuous_openapi", legacy_converter)
    monkeypatch.setattr(transform, "import_module", load_probatio)

    def serializer(schema: object) -> dict[str, str]:
        return {"type": "string"}

    tool = MagicMock(spec=llm.Tool)
    tool.name = "ping"
    tool.description = None
    tool.parameters = vol.Schema({})

    result = transform.format_tool(tool, custom_serializer=serializer)

    assert result["parameters"] == {"type": "object"}
    assert calls == [(tool.parameters, serializer)]


def test_format_tool_reports_internal_probatio_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An import failure inside probatio must not be mistaken for its absence."""
    import custom_components.codex_conversation.transform as transform

    def broken_probatio(name: str) -> ModuleType:
        raise ModuleNotFoundError(
            "No module named 'probatio.dependencies'", name="probatio.dependencies"
        )

    monkeypatch.setattr(transform, "import_module", broken_probatio)

    tool = MagicMock(spec=llm.Tool)
    tool.name = "ping"
    tool.description = None
    tool.parameters = {}

    with pytest.raises(
        HomeAssistantError,
        match="Failed to import Home Assistant's probatio schema converter",
    ) as exc_info:
        transform.format_tool(tool)

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)
