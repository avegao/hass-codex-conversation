"""Tests for OAuth token refresh integration with Home Assistant."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from homeassistant.helpers import config_entry_oauth2_flow
import pytest

from custom_components.codex_conversation.oauth import CodexOAuth2Implementation

from .conftest import TOKEN_DICT


@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        (400, config_entry_oauth2_flow.OAuth2TokenRequestReauthError),
        (401, config_entry_oauth2_flow.OAuth2TokenRequestReauthError),
        (429, config_entry_oauth2_flow.OAuth2TokenRequestTransientError),
        (500, config_entry_oauth2_flow.OAuth2TokenRequestTransientError),
    ],
)
async def test_refresh_errors_are_classified(hass, status, expected_error):
    """Refresh responses are classified for HA's reauth and retry handling."""
    response_error = aiohttp.ClientResponseError(
        request_info=MagicMock(),
        history=(),
        status=status,
        message="token refresh failed",
    )
    implementation = CodexOAuth2Implementation(hass)

    with (
        patch(
            "custom_components.codex_conversation.oauth.async_get_clientsession",
            return_value=MagicMock(),
        ),
        patch(
            "custom_components.codex_conversation.oauth.refresh_token",
            new=AsyncMock(side_effect=response_error),
        ),
    ):
        with pytest.raises(expected_error):
            await implementation._async_refresh_token(TOKEN_DICT)
