"""MaiBot 统一请求快照与 Claude Messages API 之间的映射。

只做数据转换，不发网络请求；网络调用和客户端生命周期在 plugin.py。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from anthropic import transform_schema

from .config_models import ProviderSettings

CLIENT_TYPE = "claude"
ITEM_SCHEMA_VERSION = 1
# MaiBot 任务总会给出 max_tokens；只有调用方显式传 None 时才用这个值，因为 Messages API 要求必填
DEFAULT_MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# 只有这些模型在 Claude API 上接受 fallbacks="default"；对其他模型发送该参数会被拒绝或不起作用
SERVER_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"})
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
THINKING_MODES = ("adaptive", "omit")
THINKING_DISPLAYS = ("summarized", "omitted")
JSON_OBJECT_INSTRUCTION = "只输出一个合法的 JSON 对象，不要用 Markdown 代码块包裹，也不要在 JSON 之外输出任何文字。"

_MODEL_OPTION_KEYS = frozenset({"effort", "reasoning_effort", "thinking", "thinking_display"})
# Host 会读取这两个键并改写请求顶层的 temperature / max_tokens，插件不再重复解释
_HOST_CONSUMED_KEYS = frozenset({"temperature", "max_tokens"})
# 这些 Item 来自其他 Provider 的原生输出或本插件不回放的思考内容，没有可移植的 Messages API 表示
_SKIPPED_ITEM_TYPES = frozenset({"ReasoningItem", "ProviderActivityItem", "ProviderOpaqueItem"})
_IMAGE_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}
_NORMAL_STOP_REASONS = frozenset({"end_turn", "tool_use", "stop_sequence"})


class ClaudeProviderError(RuntimeError):
    """请求无法映射为 Messages API，或响应无法交还给 MaiBot。"""


@dataclass(frozen=True, slots=True)
class ModelOptions:
    effort: str | None
    thinking: str
    thinking_display: str | None


def build_message_params(request: Mapping[str, Any], settings: ProviderSettings) -> dict[str, Any]:
    """把 MaiBot 的 response 请求快照转换为 ``client.beta.messages.stream`` 的关键字参数。"""

    model_info = _require_mapping(request.get("model_info"), "model_info")
    model = _require_text(model_info.get("model_identifier"), "model_info.model_identifier")
    options = parse_model_options(request.get("extra_params"))
    system_blocks, messages = build_conversation(request.get("context_items"))

    output_config: dict[str, Any] = {}
    format_type, schema = _read_response_format(request.get("response_format"))
    if format_type == "json_object":
        # Messages API 没有“任意 JSON 对象”模式，结构化输出要求完整 schema，只能用指令约束
        system_blocks.append({"type": "text", "text": JSON_OBJECT_INSTRUCTION})
    elif format_type == "json_schema":
        output_config["format"] = {"type": "json_schema", "schema": transform_schema(schema)}
    if options.effort is not None:
        output_config["effort"] = options.effort

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": _read_max_tokens(request.get("max_tokens")),
        "messages": messages,
    }
    cache_control = {"type": "ephemeral", "ttl": settings.cache_ttl} if settings.prompt_cache else None
    if system_blocks:
        if cache_control is not None:
            # 单独给 system 打断点：MaiBot 按窗口裁剪历史时消息前缀会变，tools+system 仍能命中
            system_blocks[-1] = {**system_blocks[-1], "cache_control": cache_control}
        params["system"] = system_blocks
    tools = build_tools(request.get("tool_options"))
    if tools:
        params["tools"] = tools
    if options.thinking == "adaptive":
        params["thinking"] = {
            "type": "adaptive",
            "display": options.thinking_display or settings.thinking_display,
        }
    if output_config:
        params["output_config"] = output_config
    if cache_control is not None:
        params["cache_control"] = cache_control
    temperature = request.get("temperature")
    if temperature is not None:
        # anthropic 1.x 去掉了 temperature 形参；仍接受它的旧模型只能经 extra_body 传入
        params["extra_body"] = {"temperature": temperature}
    if settings.refusal_fallback and model in SERVER_FALLBACK_MODELS:
        params["fallbacks"] = "default"
        params["betas"] = [FALLBACK_BETA]
    return params


def parse_model_options(raw_extra_params: Any) -> ModelOptions:
    """解析模型 ``extra_params`` 中本插件认识的键，未知键直接报错。"""

    extra_params = _require_mapping(raw_extra_params, "extra_params")
    unknown_keys = sorted(set(extra_params) - _MODEL_OPTION_KEYS - _HOST_CONSUMED_KEYS)
    if unknown_keys:
        raise ClaudeProviderError(
            f"Claude Provider 不认识这些 extra_params: {unknown_keys}；可用键为 {sorted(_MODEL_OPTION_KEYS)}"
        )

    effort = _read_choice(extra_params, "effort", EFFORT_LEVELS)
    reasoning_effort = _read_choice(extra_params, "reasoning_effort", EFFORT_LEVELS)
    if effort is not None and reasoning_effort is not None and effort != reasoning_effort:
        raise ClaudeProviderError(f"effort={effort!r} 与 reasoning_effort={reasoning_effort!r} 冲突，只保留一个")
    thinking = _read_choice(extra_params, "thinking", THINKING_MODES) or "adaptive"
    thinking_display = _read_choice(extra_params, "thinking_display", THINKING_DISPLAYS)
    if thinking == "omit" and thinking_display is not None:
        raise ClaudeProviderError("thinking='omit' 时不会发送 thinking 参数，thinking_display 无处生效")
    return ModelOptions(effort=effort or reasoning_effort, thinking=thinking, thinking_display=thinking_display)


def build_conversation(raw_items: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 Context Items 拆成顶层 system 块和 messages 列表。"""

    items = _require_sequence(raw_items, "context_items")
    system_blocks: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        where = f"context_items[{index}]"
        item = _require_mapping(raw_item, where)
        item_type = item.get("item_type")
        if item_type == "SystemMessageItem":
            blocks = _system_blocks(item.get("parts"), where)
            if messages:
                # 对话中途的 system 消息原位发送；不支持的模型会由 API 明确拒绝
                _append_message(messages, "system", blocks)
            else:
                system_blocks.extend(blocks)
        elif item_type == "UserMessageItem":
            _append_message(messages, "user", _user_blocks(item.get("parts"), where))
        elif item_type == "AssistantMessageItem":
            _append_message(messages, "assistant", _assistant_blocks(item.get("parts"), where))
        elif item_type == "FunctionCallItem":
            _append_message(messages, "assistant", [_tool_use_block(item.get("tool_call"), where)])
        elif item_type == "FunctionCallOutputItem":
            _append_message(messages, "user", [_tool_result_block(item, where)])
        elif item_type in _SKIPPED_ITEM_TYPES:
            continue
        else:
            raise ClaudeProviderError(f"{where} 的 item_type 无法转换: {item_type!r}")
    if not messages:
        raise ClaudeProviderError("请求里没有可发送的 user/assistant 消息")
    return system_blocks, messages


def build_tools(raw_tools: Any) -> list[dict[str, Any]]:
    """把 OpenAI function 结构的工具定义转换为 Messages API 工具定义。"""

    if raw_tools is None:
        return []
    tools: list[dict[str, Any]] = []
    for index, raw_tool in enumerate(_require_sequence(raw_tools, "tool_options")):
        where = f"tool_options[{index}]"
        tool = _require_mapping(raw_tool, where)
        function = _require_mapping(tool.get("function"), f"{where}.function")
        tool_payload: dict[str, Any] = {
            "name": _require_text(function.get("name"), f"{where}.function.name"),
            "input_schema": dict(_require_mapping(function.get("parameters"), f"{where}.function.parameters")),
        }
        description = function.get("description")
        if description:
            tool_payload["description"] = str(description)
        tools.append(tool_payload)
    return tools


def parse_message(message: Any) -> dict[str, Any]:
    """把 Messages API 的最终响应转换为 MaiBot 插件 Provider 的返回字典。"""

    stop_reason = message.stop_reason
    if stop_reason == "refusal":
        raise ClaudeProviderError(_describe_refusal(message))

    blocks = list(message.content)
    # 中途拒答兜底后，边界之前的工具调用与思考块已作废，只有文本作为续写上下文保留
    boundary = max((index for index, block in enumerate(blocks) if block.type == "fallback"), default=-1)
    output_items: list[dict[str, Any]] = []
    pending_text: list[str] = []

    def flush_text() -> None:
        if pending_text:
            output_items.append(
                {"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": "".join(pending_text)}]}
            )
            pending_text.clear()

    for index, block in enumerate(blocks):
        superseded = index < boundary
        if block.type == "fallback":
            continue
        if block.type == "text":
            if block.text:
                pending_text.append(block.text)
        elif block.type == "thinking":
            # display=omitted 时思考文本为空串，没有可展示的内容
            if not superseded and block.thinking:
                flush_text()
                output_items.append(
                    {
                        "item_type": "ReasoningItem",
                        "representation": "summary",
                        "summary_parts": [block.thinking],
                        "text_parts": [],
                    }
                )
        elif block.type == "redacted_thinking":
            continue
        elif block.type == "tool_use":
            if not superseded:
                flush_text()
                output_items.append(
                    {
                        "item_type": "FunctionCallItem",
                        "tool_call": {"call_id": block.id, "func_name": block.name, "args": dict(block.input)},
                    }
                )
        else:
            raise ClaudeProviderError(f"响应中出现本插件未启用的内容块类型: {block.type}")
    flush_text()

    if stop_reason == "max_tokens":
        if any(item["item_type"] == "FunctionCallItem" for item in output_items):
            raise ClaudeProviderError(
                "输出在工具调用中途达到 max_tokens 上限，参数不完整，请调大该任务或模型的 max_tokens"
            )
        if not any(item["item_type"] == "AssistantMessageItem" for item in output_items):
            raise ClaudeProviderError("max_tokens 在产出正文前就被思考耗尽，请调大该任务或模型的 max_tokens")
    elif stop_reason not in _NORMAL_STOP_REASONS:
        raise ClaudeProviderError(f"未处理的 stop_reason: {stop_reason!r}")

    return {
        "item_schema_version": ITEM_SCHEMA_VERSION,
        "output_items": output_items,
        "usage": _usage_record(message.usage),
        "response_id": message.id,
        "status": "completed",
        "raw_data": {
            "id": message.id,
            "model": message.model,
            "stop_reason": stop_reason,
            "served_by_fallback": served_by_fallback(message),
        },
    }


def served_by_fallback(message: Any) -> bool:
    """是否由兜底模型给出了最终回复；粘滞路由的轮次没有 fallback 块，只能看 usage.iterations。"""

    iterations = message.usage.iterations
    if iterations is None:
        return False
    return any(iteration.type == "fallback_message" for iteration in iterations)


def _usage_record(usage: Any) -> dict[str, int]:
    # MaiBot 的 prompt_tokens 是完整输入量，命中 + 未命中 = prompt_tokens；Claude 的 input_tokens 只含末个断点之后的部分
    creation = usage.cache_creation_input_tokens
    read = usage.cache_read_input_tokens
    prompt_tokens = usage.input_tokens + sum(value for value in (creation, read) if value is not None)
    record = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": prompt_tokens + usage.output_tokens,
    }
    if creation is not None and read is not None:
        record["prompt_cache_hit_tokens"] = read
        record["prompt_cache_miss_tokens"] = usage.input_tokens + creation
    return record


def _describe_refusal(message: Any) -> str:
    details = message.stop_details
    if details is None:
        return "Claude 因安全策略拒绝回答，响应未附带 stop_details"
    summary = f"Claude 因安全策略拒绝回答（类别: {details.category}）"
    if details.explanation:
        summary += f"：{details.explanation}"
    if details.recommended_model:
        summary += f"；API 建议改用 {details.recommended_model}"
    return summary


def _read_response_format(raw_format: Any) -> tuple[str | None, dict[str, Any] | None]:
    if raw_format is None:
        return None, None
    response_format = _require_mapping(raw_format, "response_format")
    format_type = response_format.get("format_type")
    if format_type == "text":
        return None, None
    if format_type == "json_object":
        return "json_object", None
    if format_type == "json_schema":
        wrapper = _require_mapping(response_format.get("schema"), "response_format.schema")
        return "json_schema", dict(_require_mapping(wrapper.get("schema"), "response_format.schema.schema"))
    raise ClaudeProviderError(f"不支持的 response_format.format_type: {format_type!r}")


def _read_max_tokens(raw_value: Any) -> int:
    if raw_value is None:
        return DEFAULT_MAX_TOKENS
    if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value <= 0:
        raise ClaudeProviderError(f"max_tokens 必须是正整数，实际为 {raw_value!r}")
    return raw_value


def _append_message(messages: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    # 连续同角色合并成一条，保证 tool_use 与 tool_result 各自落在相邻的单条消息里
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"].extend(blocks)
    else:
        messages.append({"role": role, "content": list(blocks)})


def _system_blocks(raw_parts: Any, where: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in _parts(raw_parts, where):
        if part.get("type") != "text":
            raise ClaudeProviderError(f"{where} 是 system 消息，只能包含文本，实际出现 {part.get('type')!r}")
        blocks.append({"type": "text", "text": _require_text(part.get("text"), f"{where}.text")})
    return blocks


def _user_blocks(raw_parts: Any, where: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in _parts(raw_parts, where):
        part_type = part.get("type")
        if part_type == "text":
            blocks.append({"type": "text", "text": _require_text(part.get("text"), f"{where}.text")})
        elif part_type == "image":
            blocks.append(_image_block(part, where))
        else:
            raise ClaudeProviderError(f"{where} 的用户消息片段类型无法转换: {part_type!r}")
    return blocks


def _assistant_blocks(raw_parts: Any, where: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in _parts(raw_parts, where):
        part_type = part.get("type")
        if part_type == "text":
            blocks.append({"type": "text", "text": _require_text(part.get("text"), f"{where}.text")})
        elif part_type == "refusal":
            blocks.append({"type": "text", "text": _require_text(part.get("refusal"), f"{where}.refusal")})
        else:
            raise ClaudeProviderError(f"{where} 是 assistant 消息，Messages API 不接受其中的 {part_type!r} 片段")
    return blocks


def _image_block(part: Mapping[str, Any], where: str) -> dict[str, Any]:
    image_format = str(part.get("image_format") or "").lower()
    media_type = _IMAGE_MEDIA_TYPES.get(image_format)
    if media_type is None:
        raise ClaudeProviderError(f"{where} 的图片格式 {image_format!r} 不受 Claude 支持，可用 jpeg/png/gif/webp")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": _require_text(part.get("image_base64"), f"{where}.image_base64"),
        },
    }


def _tool_use_block(raw_tool_call: Any, where: str) -> dict[str, Any]:
    tool_call = _require_mapping(raw_tool_call, f"{where}.tool_call")
    return {
        "type": "tool_use",
        "id": _require_text(tool_call.get("call_id"), f"{where}.tool_call.call_id"),
        "name": _require_text(tool_call.get("func_name"), f"{where}.tool_call.func_name"),
        "input": dict(_require_mapping(tool_call.get("args"), f"{where}.tool_call.args")),
    }


def _tool_result_block(item: Mapping[str, Any], where: str) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": _require_text(item.get("call_id"), f"{where}.call_id"),
    }
    output = item.get("output")
    if not isinstance(output, str):
        raise ClaudeProviderError(f"{where}.output 必须是字符串，实际为 {type(output).__name__}")
    if output:
        block["content"] = output
    if item.get("success") is False:
        block["is_error"] = True
    return block


def _parts(raw_parts: Any, where: str) -> list[Mapping[str, Any]]:
    return [
        _require_mapping(part, f"{where}.parts[{index}]")
        for index, part in enumerate(_require_sequence(raw_parts, f"{where}.parts"))
    ]


def _read_choice(source: Mapping[str, Any], key: str, choices: Sequence[str]) -> str | None:
    if key not in source:
        return None
    value = source[key]
    if value not in choices:
        raise ClaudeProviderError(f"extra_params.{key} 必须是 {list(choices)} 之一，实际为 {value!r}")
    return value


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaudeProviderError(f"{where} 必须是字典，实际为 {type(value).__name__}")
    return value


def _require_sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ClaudeProviderError(f"{where} 必须是列表，实际为 {type(value).__name__}")
    return value


def _require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ClaudeProviderError(f"{where} 必须是非空字符串，实际为 {value!r}")
    return value
