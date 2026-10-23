"""用真实 anthropic SDK + 内存 HTTP 传输验证线路格式：请求体、请求头、SSE 流解析，全程不联网。"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from support import make_provider, make_request, text_item

HOST_SECRET = "sk-ant-host-env-secret"

_SSE_EVENTS: list[dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_wire",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 30,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 200,
            },
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "先查天气"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "稍等"}},
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "toolu_w", "name": "weather", "input": {}},
    },
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"上海"}'}},
    {"type": "content_block_stop", "index": 2},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 15},
    },
    {"type": "message_stop"},
]


def _sse_body() -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in _SSE_EVENTS
    ).encode("utf-8")


@pytest.fixture
def plugin(plugin_module):
    instance = plugin_module.create_plugin()
    instance.set_plugin_config({})
    return instance


async def test_real_sdk_wire_request_and_stream_parsing(plugin) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())

    client = AsyncAnthropic(
        api_key="sk-ant-test-placeholder",
        base_url="https://api.anthropic.com",
        max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    plugin._get_client = lambda provider: client
    request = make_request(
        [text_item("SystemMessageItem", "你是麦麦"), text_item("UserMessageItem", "上海天气")],
        temperature=0.5,
        extra_params={"effort": "high"},
    )
    request["api_provider"] = make_provider(timeout=30)

    result = await plugin.handle_llm_request(operation="response", request=request)

    assert captured["url"].startswith("https://api.anthropic.com/v1/messages")
    # anthropic 2.x 默认走 API 版本 2026-10-21，Key 改放在 Authorization: Bearer
    assert captured["headers"]["anthropic-version"] == "2026-10-21"
    assert captured["headers"]["authorization"] == "Bearer sk-ant-test-placeholder"
    assert "x-api-key" not in captured["headers"]
    assert "server-side-fallback-2026-07-01" in captured["headers"]["anthropic-beta"]
    body = captured["body"]
    assert body["stream"] is True
    assert body["model"] == "claude-opus-5-5"
    assert "cache_control" not in body
    assert body["system"] == [{"type": "text", "text": "你是麦麦", "cache_control": {"type": "ephemeral", "ttl": "5m"}}]
    assert body["fallbacks"] == "default"
    assert body["temperature"] == 0.5
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert "betas" not in body

    assert result["output_items"] == [
        {"item_type": "ReasoningItem", "representation": "summary", "summary_parts": ["先查天气"], "text_parts": []},
        {"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": "稍等"}]},
        {
            "item_type": "FunctionCallItem",
            "tool_call": {"call_id": "toolu_w", "func_name": "weather", "args": {"city": "上海"}},
        },
    ]
    assert result["usage"] == {
        "prompt_tokens": 230,
        "completion_tokens": 15,
        "total_tokens": 245,
        "prompt_cache_hit_tokens": 200,
        "prompt_cache_miss_tokens": 30,
    }
    await client.close()


async def test_workspace_header_from_provider_reaches_the_wire(plugin, plugin_module, monkeypatch) -> None:
    # beta.messages.stream 自带 workspace_id 参数，未传时以 Omit 占位；要确认它不会冲掉 Provider 配的头
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["headers"] = dict(request.headers)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())

    real_client = plugin_module.AsyncAnthropic

    def client_with_mock_transport(**kwargs: Any) -> AsyncAnthropic:
        return real_client(**kwargs, http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))

    monkeypatch.setattr(plugin_module, "AsyncAnthropic", client_with_mock_transport)
    request = make_request([text_item("UserMessageItem", "hi")])
    request["api_provider"] = make_provider(
        default_headers={"anthropic-workspace-id": "wrkspc_01ExampleWorkspaceId0000"}, timeout=30
    )

    await plugin.handle_llm_request(operation="response", request=request)

    assert captured["headers"]["anthropic-workspace-id"] == "wrkspc_01ExampleWorkspaceId0000"
    await plugin.on_unload()


async def test_configured_beta_header_survives_fallback_betas(plugin, plugin_module, monkeypatch) -> None:
    # 启用兜底时 SDK 用请求级 betas 重写 anthropic-beta，Provider 里配的 beta 必须一并带上
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["headers"] = dict(request.headers)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())

    real_client = plugin_module.AsyncAnthropic

    def client_with_mock_transport(**kwargs: Any) -> AsyncAnthropic:
        return real_client(**kwargs, http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))

    monkeypatch.setattr(plugin_module, "AsyncAnthropic", client_with_mock_transport)
    request = make_request([text_item("UserMessageItem", "hi")])
    request["api_provider"] = make_provider(
        default_headers={"anthropic-beta": "context-1m-2025-08-07, files-api-2025-04-14"}, timeout=30
    )

    await plugin.handle_llm_request(operation="response", request=request)

    sent = [beta.strip() for beta in captured["headers"]["anthropic-beta"].split(",")]
    assert sent == ["context-1m-2025-08-07", "files-api-2025-04-14", "server-side-fallback-2026-07-01"]
    await plugin.on_unload()


@pytest.mark.parametrize(
    ("auth", "expected_headers", "expected_query"),
    [
        ({}, {"authorization": "Bearer sk-relay-key"}, {}),
        (
            {"auth_type": "header", "auth_header_name": "x-api-key", "auth_header_prefix": ""},
            {"x-api-key": "sk-relay-key"},
            {},
        ),
        (
            {"auth_type": "header", "auth_header_name": "Authorization", "auth_header_prefix": "Token"},
            {"authorization": "Token sk-relay-key"},
            {},
        ),
        ({"auth_type": "bearer", "auth_header_name": "api-key"}, {"api-key": "Bearer sk-relay-key"}, {}),
        ({"auth_type": "query", "auth_query_name": "key"}, {}, {"key": "sk-relay-key"}),
        ({"auth_type": "none", "api_key": ""}, {}, {}),
    ],
    ids=["bearer", "x-api-key", "custom-prefix", "custom-bearer-header", "query", "none"],
)
async def test_compatible_endpoint_auth_follows_provider_settings(
    plugin, plugin_module, monkeypatch, auth, expected_headers, expected_query
) -> None:
    # 宿主环境里的 ANTHROPIC_API_KEY 只属于本机，任何鉴权方式都不能让 SDK 把它带给第三方端点
    monkeypatch.setenv("ANTHROPIC_API_KEY", HOST_SECRET)
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = request.url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())

    real_client = plugin_module.AsyncAnthropic

    def client_with_mock_transport(**kwargs: Any) -> AsyncAnthropic:
        return real_client(**kwargs, http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))

    monkeypatch.setattr(plugin_module, "AsyncAnthropic", client_with_mock_transport)
    request = make_request([text_item("UserMessageItem", "hi")])
    request["api_provider"] = make_provider(
        **{"base_url": "https://relay.example.com/anthropic", "api_key": "sk-relay-key", "timeout": 30, **auth}
    )

    await plugin.handle_llm_request(operation="response", request=request)

    url = captured["url"]
    assert f"{url.scheme}://{url.host}{url.path}" == "https://relay.example.com/anthropic/v1/messages"
    query = dict(url.params)
    query.pop("beta")
    assert query == expected_query
    credential_headers = {
        name: value for name, value in captured["headers"].items() if name in {"authorization", "x-api-key", "api-key"}
    }
    assert credential_headers == expected_headers
    assert all(HOST_SECRET not in value for value in captured["headers"].values())
    # 兼容端点不认识服务端拒答兜底，同名模型也不发
    assert "fallbacks" not in captured["body"]
    assert "anthropic-beta" not in captured["headers"]
    await plugin.on_unload()


async def test_api_version_can_be_pinned_per_provider(plugin, plugin_module, monkeypatch) -> None:
    # 只认旧版本号的兼容端点可在 Provider 的 default_headers 里写回 2023-06-01
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["headers"] = dict(request.headers)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())

    real_client = plugin_module.AsyncAnthropic

    def client_with_mock_transport(**kwargs: Any) -> AsyncAnthropic:
        return real_client(**kwargs, http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))

    monkeypatch.setattr(plugin_module, "AsyncAnthropic", client_with_mock_transport)
    request = make_request([text_item("UserMessageItem", "hi")])
    request["api_provider"] = make_provider(
        base_url="https://relay.example.com", default_headers={"anthropic-version": "2023-06-01"}, timeout=30
    )

    await plugin.handle_llm_request(operation="response", request=request)

    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    await plugin.on_unload()
