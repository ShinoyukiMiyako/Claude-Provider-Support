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


def _block_breakpoints(params: dict) -> list[tuple[int, int]]:
    return [
        (message_index, block_index)
        for message_index, message in enumerate(params["messages"])
        for block_index, block in enumerate(message["content"])
        if "cache_control" in block
    ]


def test_history_breakpoint_lands_before_per_request_tail(adapter, settings) -> None:
    # MaiBot 的形状：历史聊天 + 机器人上一轮回复，末尾再拼本轮才有的时间与回复方向
    items = [
        text_item("SystemMessageItem", "你是白洲梓"),
        text_item("UserMessageItem", "通宵喵"),
        text_item("AssistantMessageItem", "好家伙"),
        text_item("UserMessageItem", "多刷回手机"),
        text_item("UserMessageItem", "当前时间：05:03:16"),
        text_item("UserMessageItem", "回复方向：跟着笑一下"),
    ]

    params = adapter.build_message_params(make_request(items), settings)

    assert _block_breakpoints(params) == [(1, 0)]
    assert params["messages"][1]["content"][0] == {
        "type": "text",
        "text": "好家伙",
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }
    # system + 历史 + 请求级自动断点，不超过 API 的 4 个上限
    assert params["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_history_breakpoint_uses_tool_result_when_it_is_the_latest_model_side_block(adapter, settings) -> None:
    params = adapter.build_message_params(make_request(_tool_round_items()), settings)

    # 最后一块模型侧内容是 call_b 的结果，后面的“继续”留给请求级自动断点
    assert _block_breakpoints(params) == [(2, 1)]
    assert params["messages"][2]["content"][1]["tool_use_id"] == "call_b"


def test_no_history_breakpoint_when_model_side_block_is_already_last(adapter, settings) -> None:
    items = [*_tool_round_items()[:-1]]

    params = adapter.build_message_params(make_request(items), settings)

    assert _block_breakpoints(params) == []
    assert params["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_no_history_breakpoint_without_model_side_blocks(adapter, settings) -> None:
    items = [text_item("SystemMessageItem", "你是白洲梓"), text_item("UserMessageItem", "在吗")]

    params = adapter.build_message_params(make_request(items), settings)

    assert _block_breakpoints(params) == []


def test_split_message_parts_are_joined_back_into_one_block(adapter) -> None:
    item = {
        "item_type": "UserMessageItem",
        "meta": meta(),
        "parts": [
            {"type": "text", "text": "哦耶耶～桃花"},
            {"type": "text", "text": "🌸"},
            {"type": "text", "text": "谢了有玫瑰"},
            {"type": "image", "image_format": "png", "image_base64": "QUJD"},
            {"type": "text", "text": "看图"},
        ],
    }

    _, messages = adapter.build_conversation([item])

    assert [block["type"] for block in messages[0]["content"]] == ["text", "image", "text"]
    assert messages[0]["content"][0]["text"] == "哦耶耶～桃花🌸谢了有玫瑰"
    assert messages[0]["content"][2]["text"] == "看图"


def test_consecutive_user_messages_share_one_text_block(adapter) -> None:
    items = [
        text_item("UserMessageItem", "第一条"),
        text_item("UserMessageItem", "第二条"),
        text_item("AssistantMessageItem", "收到"),
        *_tool_round_items()[4:8],
        text_item("UserMessageItem", "第三条"),
        text_item("UserMessageItem", "第四条"),
    ]

    _, messages = adapter.build_conversation(items)

    assert messages[0]["content"] == [{"type": "text", "text": "第一条\n第二条"}]
    # 工具结果不与文字合并，结果之后的文字仍合成一块
    assert [block["type"] for block in messages[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert messages[2]["content"][2]["text"] == "第三条\n第四条"


def test_message_burst_stays_within_cache_lookback(adapter, settings) -> None:
    # 两轮之间群友连发 30 条：下一轮的历史断点到上一轮断点之间仍只隔几个块，不超出 20 块回看范围
    burst = [text_item("UserMessageItem", f"第 {index} 条") for index in range(30)]
    items = [
        text_item("SystemMessageItem", "你是白洲梓"),
        text_item("AssistantMessageItem", "上一轮回复"),
        *burst,
        text_item("AssistantMessageItem", "这一轮回复"),
        text_item("UserMessageItem", "当前时间：05:03"),
    ]

    params = adapter.build_message_params(make_request(items), settings)

    flat = [block for message in params["messages"] for block in message["content"]]
    previous_anchor = next(i for i, block in enumerate(flat) if block.get("text") == "上一轮回复")
    current_anchor = next(i for i, block in enumerate(flat) if "cache_control" in block)
    assert current_anchor - previous_anchor <= 20


def test_cache_disabled_leaves_no_breakpoints(adapter, config_models) -> None:
    settings = config_models.ProviderSettings(prompt_cache=False)

    params = adapter.build_message_params(make_request(_tool_round_items()), settings)

    assert "cache_control" not in params
    assert all("cache_control" not in block for block in params["system"])
    assert _block_breakpoints(params) == []


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
