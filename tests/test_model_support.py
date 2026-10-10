"""Tests for GPT-6 selection and request serialization."""

from __future__ import annotations

from homeassistant.config_entries import ConfigSubentry
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.codex_conversation.codex_api.requests import CodexRequest
from custom_components.codex_conversation.config_flow import (
    CodexAITaskSubentryFlow,
    CodexConversationSubentryFlow,
)
from custom_components.codex_conversation.const import (
    CONF_MODEL,
    CONF_REASONING_EFFORT,
    DOMAIN,
)

GPT6_MODELS = ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol"]
SUBENTRY_FLOWS = [
    (CodexConversationSubentryFlow, "conversation"),
    (CodexAITaskSubentryFlow, "ai_task_data"),
]


@pytest.mark.parametrize("model", GPT6_MODELS)
@pytest.mark.parametrize(("flow_class", "subentry_type"), SUBENTRY_FLOWS)
async def test_gpt6_model_can_be_selected(hass, model, flow_class, subentry_type):
    """Both subentry selectors accept and save each new model."""
    flow = flow_class()
    flow.hass = hass
    flow.handler = ("test_entry", subentry_type)
    flow.context = {"source": "user"}
    flow.options = flow._default_data.copy()
    flow._init_data = {}

    form = await flow.async_step_advanced()
    data = form["data_schema"]({CONF_MODEL: model})
    result = await flow.async_step_advanced(data)

    assert result["type"] == "create_entry"
    assert result["title"] == f"Codex ({model})"
    assert result["data"][CONF_MODEL] == model
    assert result["data"][CONF_REASONING_EFFORT] == "medium"


@pytest.mark.parametrize("model", GPT6_MODELS)
@pytest.mark.parametrize(("flow_class", "subentry_type"), SUBENTRY_FLOWS)
async def test_gpt6_model_can_be_reconfigured(hass, model, flow_class, subentry_type):
    """Existing conversation and AI task subentries can switch to GPT-6."""
    subentry = ConfigSubentry(
        data=flow_class()._default_data.copy(),
        subentry_type=subentry_type,
        title="Existing Codex",
        unique_id=None,
    )
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    hass.config_entries.async_add_subentry(entry, subentry)
    flow = flow_class()
    flow.hass = hass
    flow.handler = (entry.entry_id, subentry_type)
    flow.context = {"source": "reconfigure", "subentry_id": subentry.subentry_id}
    flow.options = subentry.data.copy()
    flow._init_data = subentry.data.copy()

    form = await flow.async_step_advanced()
    data = form["data_schema"]({CONF_MODEL: model})
    result = await flow.async_step_advanced(data)

    assert result["type"] == "abort"
    assert result["reason"] == "reconfigure_successful"
    assert entry.subentries[subentry.subentry_id].data[CONF_MODEL] == model
    assert entry.subentries[subentry.subentry_id].title == f"Codex ({model})"


@pytest.mark.parametrize("model", GPT6_MODELS)
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_gpt6_request_preserves_reasoning_options(model, effort):
    """GPT-6 requests send the reasoning controls and encrypted history include."""
    request = CodexRequest(
        model=model,
        input=[],
        reasoning_effort=effort,
        reasoning_summary="detailed",
        text_verbosity="low",
    )

    body = request.to_body()

    assert body["model"] == model
    assert body["reasoning"] == {"effort": effort, "summary": "detailed"}
    assert body["text"] == {"verbosity": "low"}
    assert body["include"] == ["reasoning.encrypted_content"]


def test_non_reasoning_request_still_omits_reasoning_options():
    """The GPT-6 addition preserves serialization for non-reasoning models."""
    body = CodexRequest(model="gpt-4o", input=[]).to_body()

    assert body == {
        "model": "gpt-4o",
        "stream": True,
        "store": False,
        "input": [],
    }
