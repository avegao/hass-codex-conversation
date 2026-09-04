"""Tests for the Codex config and reauthentication flows."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.config_entries import SOURCE_REAUTH
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.codex_conversation.codex_api.auth import OAuthToken
from custom_components.codex_conversation.config_flow import (
    CodexConversationConfigFlow,
)
from custom_components.codex_conversation.const import DOMAIN


def _reauth_flow(hass, entry: MockConfigEntry) -> CodexConversationConfigFlow:
    """Build a config flow linked to an existing entry."""
    flow = CodexConversationConfigFlow()
    flow.hass = hass
    flow.context = {"source": SOURCE_REAUTH, "entry_id": entry.entry_id}
    return flow


async def test_reauth_starts_with_confirmation(hass):
    """Reauthentication must ask for confirmation before starting device auth."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="reauth_entry",
        data={"auth_implementation": DOMAIN, "token": {}},
    )
    entry.add_to_hass(hass)
    flow = _reauth_flow(hass, entry)

    result = await flow.async_step_reauth(entry.data)

    assert result["type"] == "form"
    assert result["step_id"] == "reauth_confirm"


async def test_reauth_finish_updates_existing_entry(hass):
    """A successful reauth replaces only the token on the existing entry."""
    old_token = {"access_token": "old", "refresh_token": "old-refresh"}
    entry = MockConfigEntry(
        domain=DOMAIN,
        entry_id="reauth_entry",
        data={
            "auth_implementation": DOMAIN,
            "token": old_token,
            "unrelated": "preserved",
        },
    )
    entry.add_to_hass(hass)
    flow = _reauth_flow(hass, entry)
    flow._token = OAuthToken(
        access_token="new",
        refresh_token="new-refresh",
        account_id="account",
        expires_at=2_000_000_000,
        expires_in=3600,
    )

    result = await flow.async_step_finish()

    assert result["type"] == "abort"
    assert result["reason"] == "reauth_successful"
    assert entry.data["token"] == flow._token.as_dict()
    assert entry.data["unrelated"] == "preserved"


async def test_reauth_user_step_does_not_abort_for_existing_entry(hass):
    """The setup step must be reusable by reauthentication flows."""
    entry = MockConfigEntry(domain=DOMAIN, entry_id="reauth_entry", data={})
    flow = _reauth_flow(hass, entry)
    expected = {"type": "progress", "step_id": "activate"}

    with patch.object(
        flow,
        "_async_start_device_flow",
        new=AsyncMock(return_value=expected),
    ) as start_device_flow:
        result = await flow.async_step_user()

    assert result == expected
    start_device_flow.assert_awaited_once_with()
