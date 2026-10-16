from __future__ import annotations

import pytest
from support import make_message


def test_text_thinking_and_tool_use_become_items_in_order(adapter) -> None:
    message = make_message(
        [
            {"type": "thinking", "thinking": "先确认用户要什么", "signature": "sig"},
            {"type": "text", "text": "我去查"},
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "猫"}},
        ],
        stop_reason="tool_use",
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 300,
        },
    )

    result = adapter.parse_message(message)

    assert result["item_schema_version"] == 1
    assert result["output_items"] == [
        {
            "item_type": "ReasoningItem",
            "representation": "summary",
            "summary_parts": ["先确认用户要什么"],
            "text_parts": [],
        },
        {"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": "我去查"}]},
        {
            "item_type": "FunctionCallItem",
            "tool_call": {"call_id": "toolu_1", "func_name": "search", "args": {"q": "猫"}},
        },
    ]
    # 命中 300、写入 20、末段 100：MaiBot 口径下完整输入 420，未命中 120
    assert result["usage"] == {
        "prompt_tokens": 420,
        "completion_tokens": 50,
        "total_tokens": 470,
        "prompt_cache_hit_tokens": 300,
        "prompt_cache_miss_tokens": 120,
    }
    assert result["response_id"] == "msg_test"
    assert result["raw_data"]["served_by_fallback"] is False


def test_omitted_thinking_and_redacted_blocks_are_not_emitted(adapter) -> None:
    message = make_message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "你好"},
            {"type": "text", "text": "呀"},
        ]
    )

    result = adapter.parse_message(message)

    assert result["output_items"] == [
        {"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": "你好呀"}]}
    ]


def test_fallback_boundary_drops_superseded_tool_calls_but_keeps_text(adapter) -> None:
    message = make_message(
        [
            {"type": "text", "text": "前半句，"},
            {"type": "tool_use", "id": "toolu_dead", "name": "search", "input": {}},
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5-5"},
                "to": {"model": "claude-opus-4-8"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "后半句"},
        ],
        model="claude-opus-4-8",
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "iterations": [
                {
                    "type": "message",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-4-8",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ],
        },
    )

    result = adapter.parse_message(message)

    assert result["output_items"] == [
        {"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": "前半句，后半句"}]}
    ]
    assert result["raw_data"]["served_by_fallback"] is True
    assert result["raw_data"]["model"] == "claude-opus-4-8"


def test_refusal_raises_with_category_and_explanation(adapter) -> None:
    message = make_message(
        [],
        stop_reason="refusal",
        stop_details={
            "type": "refusal",
            "category": "bio",
            "explanation": "涉及危险内容",
            "recommended_model": "claude-opus-4-8",
        },
    )

    with pytest.raises(adapter.ClaudeProviderError, match="bio.*涉及危险内容.*claude-opus-4-8"):
        adapter.parse_message(message)


def test_max_tokens_with_text_returns_partial_reply(adapter) -> None:
    result = adapter.parse_message(make_message([{"type": "text", "text": "说到一半"}], stop_reason="max_tokens"))

    assert result["output_items"][0]["parts"][0]["text"] == "说到一半"
    assert result["raw_data"]["stop_reason"] == "max_tokens"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([{"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}], "工具调用中途"),
        ([{"type": "thinking", "thinking": "想了很多", "signature": "sig"}], "正文前"),
    ],
)
def test_max_tokens_without_usable_reply_raises(adapter, content, message) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match=message):
        adapter.parse_message(make_message(content, stop_reason="max_tokens"))


def test_context_window_exceeded_with_text_returns_partial_reply(adapter) -> None:
    result = adapter.parse_message(
        make_message([{"type": "text", "text": "说到一半"}], stop_reason="model_context_window_exceeded")
    )

    assert result["output_items"][0]["parts"][0]["text"] == "说到一半"
    assert result["raw_data"]["stop_reason"] == "model_context_window_exceeded"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([{"type": "tool_use", "id": "toolu_1", "name": "search", "input": {}}], "工具调用中途.*上下文窗口"),
        ([{"type": "thinking", "thinking": "想了很多", "signature": "sig"}], "正文前.*上下文窗口"),
    ],
)
def test_context_window_exceeded_without_usable_reply_raises(adapter, content, message) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match=message):
        adapter.parse_message(make_message(content, stop_reason="model_context_window_exceeded"))


def test_unhandled_stop_reason_raises(adapter) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match="pause_turn"):
        adapter.parse_message(make_message([{"type": "text", "text": "x"}], stop_reason="pause_turn"))


@pytest.mark.parametrize(
    "content", [[], [{"type": "text", "text": " \n"}], [{"type": "thinking", "thinking": "", "signature": "sig"}]]
)
def test_empty_reply_raises_so_host_retries(adapter, content) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match="没有正文"):
        adapter.parse_message(make_message(content))


def test_blank_text_next_to_thinking_is_not_emitted(adapter) -> None:
    message = make_message(
        [{"type": "thinking", "thinking": "该不该回", "signature": "sig"}, {"type": "text", "text": "\n"}]
    )

    result = adapter.parse_message(message)

    assert [item["item_type"] for item in result["output_items"]] == ["ReasoningItem"]


def test_unexpected_block_type_raises(adapter) -> None:
    message = make_message(
        [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "x"}}]
    )

    with pytest.raises(adapter.ClaudeProviderError, match="server_tool_use"):
        adapter.parse_message(message)


def test_usage_without_cache_fields_does_not_claim_cache_stats(adapter) -> None:
    message = make_message([{"type": "text", "text": "hi"}], usage={"input_tokens": 12, "output_tokens": 3})

    result = adapter.parse_message(message)

    assert result["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
