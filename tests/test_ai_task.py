"""Tests for the AI Task entity."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

from homeassistant.components import ai_task as ai_task_component
from homeassistant.components.conversation import (
    AssistantContent,
    Attachment,
    UserContent,
)
from homeassistant.config_entries import ConfigSubentry
from homeassistant.exceptions import HomeAssistantError
import pytest
import voluptuous as vol

from custom_components.codex_conversation.ai_task import (
    CodexAITaskEntity,
    _format_structure_instruction,
)
from custom_components.codex_conversation.codex_api import OutputTextDelta
from custom_components.codex_conversation.const import DOMAIN

from .conftest import make_chat_log


@pytest.fixture
def mock_ai_task_entity(
    hass, mock_config_entry, mock_oauth_session, mock_ai_task_subentry: SimpleNamespace
) -> CodexAITaskEntity:
    """A CodexAITaskEntity wired to hass but not added to the entity registry."""
    entity = CodexAITaskEntity(
        hass,
        mock_config_entry,
        mock_oauth_session,
        cast(ConfigSubentry, mock_ai_task_subentry),
    )
    entity.entity_id = f"ai_task.{DOMAIN}"
    entity.hass = hass
    return entity


async def test_generate_data_returns_text_result(mock_ai_task_entity):
    """The entity should return plain text when no structure is requested."""
    chat_log = make_chat_log(
        [
            AssistantContent(
                agent_id="ai_task.codex", content="Result text", tool_calls=None
            )
        ]
    )
    chat_log.conversation_id = "conv-1"
    task = MagicMock(spec=ai_task_component.GenDataTask)
    task.structure = None
    task.name = "summarize"

    async def fake_stream(request):
        yield OutputTextDelta(delta="Result text", content_index=0)

    with patch(
        "custom_components.codex_conversation.ai_task.CodexClient"
    ) as MockClient:
        MockClient.return_value.stream = fake_stream
        result = await mock_ai_task_entity._async_generate_data(task, chat_log)

    assert result.conversation_id == "conv-1"
    assert result.data == "Result text"


async def test_generate_data_parses_json_result(mock_ai_task_entity):
    """Structured tasks should parse the assistant text as JSON."""
    chat_log = make_chat_log(
        [
            AssistantContent(
                agent_id="ai_task.codex", content='{"answer":"ok"}', tool_calls=None
            )
        ]
    )
    chat_log.conversation_id = "conv-2"
    task = MagicMock(spec=ai_task_component.GenDataTask)
    task.structure = vol.Schema({vol.Required("answer"): str})
    task.name = "extract"

    async def fake_stream(request):
        yield OutputTextDelta(delta='{"answer":"ok"}', content_index=0)

    with patch(
        "custom_components.codex_conversation.ai_task.CodexClient"
    ) as MockClient:
        MockClient.return_value.stream = fake_stream
        result = await mock_ai_task_entity._async_generate_data(task, chat_log)

    assert result.conversation_id == "conv-2"
    assert result.data == {"answer": "ok"}


def test_ai_task_entity_supports_attachments(mock_ai_task_entity):
    """The entity should advertise attachment support when it can replay them."""
    assert (
        mock_ai_task_entity.supported_features
        & ai_task_component.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )


async def test_generate_data_passes_attachments_to_codex_request(
    mock_ai_task_entity, tmp_path
):
    """Attachments should be forwarded via the shared chat-log path."""
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-png-data")

    chat_log = make_chat_log(
        [
            UserContent(
                content="Describe this image",
                attachments=[
                    Attachment(
                        media_content_id="media://camera/image",
                        mime_type="image/png",
                        path=image_path,
                    )
                ],
            ),
        ]
    )
    task = MagicMock(spec=ai_task_component.GenDataTask)
    task.structure = None
    task.name = "describe"

    captured_requests: list = []

    async def capturing_stream(request):
        captured_requests.append(request)
        yield OutputTextDelta(delta="Result text", content_index=0)

    async def persist_stream(entity_id, gen):
        text_chunks: list[str] = []
        async for delta in gen:
            if delta.get("content"):
                text_chunks.append(delta["content"])
        chat_log.content.append(
            AssistantContent(
                agent_id=entity_id,
                content="".join(text_chunks),
                tool_calls=None,
            )
        )
        return
        yield

    chat_log.async_add_delta_content_stream = persist_stream

    with patch(
        "custom_components.codex_conversation.ai_task.CodexClient"
    ) as MockClient:
        MockClient.return_value.stream = capturing_stream
        await mock_ai_task_entity._async_generate_data(task, chat_log)

    assert len(captured_requests) == 1
    content = captured_requests[0].input[-1]["content"]
    assert any(item["type"] == "input_image" for item in content)


async def test_ai_task_rejects_attachments_when_entity_does_not_support_them(hass):
    """Home Assistant should reject attachments before dispatch when the feature is absent."""
    from homeassistant.components.ai_task import task as ai_task_module

    entity = MagicMock()
    entity.supported_features = ai_task_component.AITaskEntityFeature.GENERATE_DATA

    hass.data[ai_task_module.DATA_COMPONENT] = SimpleNamespace(
        get_entity=lambda entity_id: entity
    )

    with pytest.raises(HomeAssistantError, match="does not support attachments"):
        await ai_task_module.async_generate_data(
            hass,
            task_name="describe",
            entity_id="ai_task.test",
            instructions="Describe this image",
            attachments=[{"path": "/tmp/sample.png", "mime_type": "image/png"}],
        )


def test_format_structure_instruction():
    """The structured output helper should request JSON-only output."""
    task = MagicMock(spec=ai_task_component.GenDataTask)
    task.structure = vol.Schema({vol.Required("name"): str, vol.Optional("age"): int})

    instruction = _format_structure_instruction(task)

    assert "Return only valid JSON." in instruction
    assert "name" in instruction
    assert "age" in instruction
