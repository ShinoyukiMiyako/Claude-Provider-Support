"""用真实 anthropic SDK + 内存 HTTP 传输验证线路格式：请求体、请求头、SSE 流解析，全程不联网。"""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient
from support import make_request, text_item

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
    request["api_provider"] = {"timeout": 30}

    result = await plugin.handle_llm_request(operation="response", request=request)

    assert captured["url"].startswith("https://api.anthropic.com/v1/messages")
    assert captured["headers"]["x-api-key"] == "sk-ant-test-placeholder"
    assert "server-side-fallback-2026-07-01" in captured["headers"]["anthropic-beta"]
    body = captured["body"]
    assert body["stream"] is True
    assert body["model"] == "claude-opus-5-5"
    assert body["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
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
    request["api_provider"] = {
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-ant-test-placeholder",
        "auth_type": "bearer",
        "default_headers": {"anthropic-workspace-id": "wrkspc_01ExampleWorkspaceId0000"},
        "default_query": {},
        "timeout": 30,
    }

    await plugin.handle_llm_request(operation="response", request=request)

    assert captured["headers"]["anthropic-workspace-id"] == "wrkspc_01ExampleWorkspaceId0000"
    await plugin.on_unload()
