from __future__ import annotations

import json
import tomllib
from typing import Any

import pytest
from support import PLUGIN_DIR, FakeClient, FakeStream, make_message, make_request, text_item


def _provider(**overrides: Any) -> dict[str, Any]:
    provider: dict[str, Any] = {
        "name": "Claude",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-ant-test-placeholder",
        "client_type": "claude",
        "auth_type": "bearer",
        "auth_header_name": "Authorization",
        "auth_header_prefix": "Bearer",
        "auth_query_name": "api_key",
        "default_headers": {},
        "default_query": {},
        "max_retry": 3,
        "timeout": 120,
        "retry_interval": 4,
    }
    provider.update(overrides)
    return provider


@pytest.fixture
def plugin(plugin_module):
    instance = plugin_module.create_plugin()
    instance.set_plugin_config({})
    return instance


async def test_handler_sends_mapped_params_and_returns_items(plugin) -> None:
    fake = FakeClient(FakeStream(make_message([{"type": "text", "text": "喵"}])))
    plugin._get_client = lambda provider: fake
    request = make_request([text_item("SystemMessageItem", "你是猫"), text_item("UserMessageItem", "叫一声")])
    request["api_provider"] = _provider()

    result = await plugin.handle_llm_request(operation="response", request=request)

    assert result["output_items"] == [{"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": "喵"}]}]
    sent = fake.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["messages"] == [{"role": "user", "content": [{"type": "text", "text": "叫一声"}]}]
    assert sent["system"][0]["text"] == "你是猫"


async def test_total_timeout_aborts_stream(plugin, adapter) -> None:
    stream = FakeStream(make_message([{"type": "text", "text": "迟到"}]), delay=1.0)
    plugin._get_client = lambda provider: FakeClient(stream)
    request = make_request([text_item("UserMessageItem", "hi")])
    request["api_provider"] = _provider(timeout=0.05)

    with pytest.raises(adapter.ClaudeProviderError, match="0.05s"):
        await plugin.handle_llm_request(operation="response", request=request)

    assert stream.exited is True


@pytest.mark.parametrize("operation", ["embedding", "audio_transcription", "image_embedding"])
async def test_non_chat_operations_are_rejected(plugin, adapter, operation) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match=operation):
        await plugin.handle_llm_request(operation=operation, request={})


async def test_clients_are_reused_per_provider_and_closed_on_unload(plugin) -> None:
    first = plugin._get_client(_provider())
    same = plugin._get_client(_provider())
    other = plugin._get_client(_provider(api_key="sk-ant-other-placeholder"))

    assert first is same
    assert other is not first
    assert str(first.base_url).rstrip("/") == "https://api.anthropic.com"
    assert first.max_retries == 0

    await plugin.on_unload()

    assert first.is_closed() and other.is_closed()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": "https://api.anthropic.com/v1"}, "不要带 /v1"),
        ({"base_url": "api.anthropic.com"}, "http"),
        ({"base_url": ""}, "base_url 为空"),
        ({"auth_type": "header"}, "x-api-key"),
        ({"api_key": ""}, "api_key 为空"),
        ({"default_headers": {"anthropic-workspace-id": " wrkspc_01ExampleWorkspaceId0000"}}, "wrkspc_"),
        ({"default_headers": {"Anthropic-Workspace-Id": "Default"}}, "Anthropic-Workspace-Id"),
    ],
)
def test_invalid_provider_config_is_rejected(plugin, adapter, overrides, message) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match=message):
        plugin._get_client(_provider(**overrides))


def test_manifest_code_and_pyproject_agree(plugin, plugin_module) -> None:
    manifest = json.loads((PLUGIN_DIR / "_manifest.json").read_text(encoding="utf-8"))
    pyproject = tomllib.loads((PLUGIN_DIR / "pyproject.toml").read_text(encoding="utf-8"))

    declared = [provider["client_type"] for provider in manifest["llm_providers"]]
    registered = [provider["client_type"] for provider in plugin.get_llm_providers()]
    assert declared == registered == ["claude"]
    assert manifest["version"] == plugin_module.PLUGIN_VERSION == pyproject["project"]["version"]
    manifest_anthropic = next(dep for dep in manifest["dependencies"] if dep["name"] == "anthropic")
    assert f"anthropic{manifest_anthropic['version_spec']}" in pyproject["project"]["dependencies"]
