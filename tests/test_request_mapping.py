from __future__ import annotations

import pytest
from support import make_request, meta, text_item


def _tool_round_items() -> list[dict]:
    return [
        text_item("SystemMessageItem", "你是麦麦"),
        {
            "item_type": "UserMessageItem",
            "meta": meta(),
            "parts": [
                {"type": "text", "text": "看看这张图"},
                {"type": "image", "image_format": "jpg", "image_base64": "QUJD"},
            ],
        },
        {
            "item_type": "ReasoningItem",
            "meta": meta("t1"),
            "representation": "raw_text",
            "summary_parts": [],
            "text_parts": ["上一轮的思考"],
        },
        text_item("AssistantMessageItem", "我查一下"),
        {
            "item_type": "FunctionCallItem",
            "meta": meta("t1"),
            "tool_call": {"call_id": "call_a", "func_name": "search", "args": {"q": "猫"}, "extra_content": None},
        },
        {
            "item_type": "FunctionCallItem",
            "meta": meta("t1"),
            "tool_call": {"call_id": "call_b", "func_name": "weather", "args": {}, "extra_content": None},
        },
        {
            "item_type": "FunctionCallOutputItem",
            "meta": meta("t1"),
            "call_id": "call_a",
            "output": "找到 3 条",
            "success": True,
            "tool_name": "search",
        },
        {
            "item_type": "FunctionCallOutputItem",
            "meta": meta("t1"),
            "call_id": "call_b",
            "output": "接口超时",
            "success": False,
            "tool_name": "weather",
        },
        text_item("UserMessageItem", "继续"),
    ]


def test_conversation_maps_roles_tools_and_images(adapter) -> None:
    system_blocks, messages = adapter.build_conversation(_tool_round_items())

    assert system_blocks == [{"type": "text", "text": "你是麦麦"}]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
    }
    # 思考内容不回放；文本与两次工具调用落在同一条 assistant 消息里
    assert messages[1]["content"] == [
        {"type": "text", "text": "我查一下"},
        {"type": "tool_use", "id": "call_a", "name": "search", "input": {"q": "猫"}},
        {"type": "tool_use", "id": "call_b", "name": "weather", "input": {}},
    ]
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_a", "content": "找到 3 条"},
        {"type": "tool_result", "tool_use_id": "call_b", "content": "接口超时", "is_error": True},
        {"type": "text", "text": "继续"},
    ]


def test_mid_conversation_system_message_stays_in_place(adapter) -> None:
    system_blocks, messages = adapter.build_conversation(
        [
            text_item("SystemMessageItem", "全局设定"),
            text_item("UserMessageItem", "你好"),
            text_item("SystemMessageItem", "接下来简短回答"),
        ]
    )

    assert system_blocks == [{"type": "text", "text": "全局设定"}]
    assert messages[-1] == {"role": "system", "content": [{"type": "text", "text": "接下来简短回答"}]}


def test_empty_tool_output_omits_content(adapter) -> None:
    _, messages = adapter.build_conversation(
        [
            text_item("UserMessageItem", "执行"),
            {
                "item_type": "FunctionCallItem",
                "meta": meta("t1"),
                "tool_call": {"call_id": "call_a", "func_name": "noop", "args": {}, "extra_content": None},
            },
            {
                "item_type": "FunctionCallOutputItem",
                "meta": meta("t1"),
                "call_id": "call_a",
                "output": "",
                "success": True,
                "tool_name": "noop",
            },
        ]
    )

    assert messages[-1]["content"] == [{"type": "tool_result", "tool_use_id": "call_a"}]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (
            [
                {
                    "item_type": "AssistantMessageItem",
                    "meta": meta(),
                    "parts": [{"type": "image", "image_format": "png", "image_base64": "QUJD"}],
                }
            ],
            "assistant",
        ),
        (
            [
                {
                    "item_type": "UserMessageItem",
                    "meta": meta(),
                    "parts": [{"type": "image", "image_format": "bmp", "image_base64": "QUJD"}],
                }
            ],
            "bmp",
        ),
        ([{"item_type": "MysteryItem", "meta": meta()}], "MysteryItem"),
        ([text_item("SystemMessageItem", "只有系统提示")], "user/assistant"),
    ],
)
def test_unmappable_conversation_is_rejected(adapter, items, message) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match=message):
        adapter.build_conversation(items)


def test_default_params_enable_cache_thinking_and_fallback(adapter, settings) -> None:
    params = adapter.build_message_params(make_request(_tool_round_items()), settings)

    assert params["model"] == "claude-opus-5-5"
    assert params["max_tokens"] == 4096
    assert params["system"] == [
        {"type": "text", "text": "你是麦麦", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
    ]
    assert params["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert params["fallbacks"] == "default"
    assert params["betas"] == ["server-side-fallback-2026-07-01"]
    assert "extra_body" not in params
    assert "output_config" not in params


def test_cache_disabled_leaves_no_breakpoints(adapter, config_models) -> None:
    settings = config_models.ProviderSettings(prompt_cache=False)

    params = adapter.build_message_params(make_request(_tool_round_items()), settings)

    assert "cache_control" not in params
    assert all("cache_control" not in block for block in params["system"])


def test_one_hour_ttl_applies_to_both_breakpoints(adapter, config_models) -> None:
    settings = config_models.ProviderSettings(cache_ttl="1h")

    params = adapter.build_message_params(make_request(_tool_round_items()), settings)

    assert params["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert params["system"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-opus-4-8", "claude-sonnet-4-6"])
def test_fallback_only_sent_to_supporting_models(adapter, settings, model) -> None:
    request = make_request([text_item("UserMessageItem", "hi")])
    request["model_info"]["model_identifier"] = model

    params = adapter.build_message_params(request, settings)

    assert "fallbacks" not in params
    assert "betas" not in params


def test_fallback_can_be_disabled(adapter, config_models) -> None:
    settings = config_models.ProviderSettings(refusal_fallback=False)

    params = adapter.build_message_params(make_request([text_item("UserMessageItem", "hi")]), settings)

    assert "fallbacks" not in params


def test_temperature_is_forwarded_through_extra_body(adapter, settings) -> None:
    params = adapter.build_message_params(make_request([text_item("UserMessageItem", "hi")], temperature=0.7), settings)

    assert params["extra_body"] == {"temperature": 0.7}


def test_json_object_appends_instruction_as_cached_last_system_block(adapter, settings) -> None:
    request = make_request(
        [text_item("SystemMessageItem", "评分器"), text_item("UserMessageItem", "打分")],
        response_format={"format_type": "json_object"},
    )

    params = adapter.build_message_params(request, settings)

    assert [block["text"] for block in params["system"]] == ["评分器", adapter.JSON_OBJECT_INSTRUCTION]
    assert "cache_control" not in params["system"][0]
    assert params["system"][1]["cache_control"]["type"] == "ephemeral"


def test_json_schema_is_transformed_for_structured_outputs(adapter, settings) -> None:
    request = make_request(
        [text_item("UserMessageItem", "打分")],
        response_format={
            "format_type": "json_schema",
            "schema": {
                "name": "Score",
                "schema": {
                    "type": "object",
                    "properties": {"score": {"type": "integer", "minimum": 0}},
                    "required": ["score"],
                },
                "strict": False,
            },
        },
    )

    params = adapter.build_message_params(request, settings)

    output_format = params["output_config"]["format"]
    assert output_format["type"] == "json_schema"
    assert output_format["schema"]["additionalProperties"] is False
    assert output_format["schema"]["required"] == ["score"]
    # 结构化输出不支持 minimum，SDK 把它挪进描述
    assert "minimum" not in output_format["schema"]["properties"]["score"]
    assert "minimum: 0" in output_format["schema"]["properties"]["score"]["description"]


def test_effort_and_thinking_options(adapter, settings) -> None:
    request = make_request(
        [text_item("UserMessageItem", "hi")],
        extra_params={"reasoning_effort": "xhigh", "thinking_display": "omitted", "temperature": 1, "max_tokens": 9},
    )

    params = adapter.build_message_params(request, settings)

    assert params["output_config"] == {"effort": "xhigh"}
    assert params["thinking"] == {"type": "adaptive", "display": "omitted"}


def test_thinking_omit_sends_no_thinking_param(adapter, settings) -> None:
    request = make_request([text_item("UserMessageItem", "hi")], extra_params={"thinking": "omit"})

    params = adapter.build_message_params(request, settings)

    assert "thinking" not in params


@pytest.mark.parametrize(
    ("extra_params", "message"),
    [
        ({"reasoning": {"effort": "high"}}, "不认识"),
        ({"effort": "extreme"}, "effort"),
        ({"effort": "low", "reasoning_effort": "high"}, "冲突"),
        ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, "thinking"),
        ({"thinking": "omit", "thinking_display": "summarized"}, "thinking_display"),
    ],
)
def test_invalid_model_options_are_rejected(adapter, extra_params, message) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match=message):
        adapter.parse_model_options(extra_params)


def test_tools_use_input_schema(adapter, settings) -> None:
    request = make_request(
        [text_item("UserMessageItem", "hi")],
        tool_options=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "搜索",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                },
            }
        ],
    )

    params = adapter.build_message_params(request, settings)

    assert params["tools"] == [
        {
            "name": "search",
            "description": "搜索",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }
    ]


def test_missing_max_tokens_uses_documented_default(adapter, settings) -> None:
    params = adapter.build_message_params(make_request([text_item("UserMessageItem", "hi")], max_tokens=None), settings)

    assert params["max_tokens"] == adapter.DEFAULT_MAX_TOKENS


@pytest.mark.parametrize("value", [0, -1, True, "4096"])
def test_invalid_max_tokens_is_rejected(adapter, settings, value) -> None:
    with pytest.raises(adapter.ClaudeProviderError, match="max_tokens"):
        adapter.build_message_params(make_request([text_item("UserMessageItem", "hi")], max_tokens=value), settings)
