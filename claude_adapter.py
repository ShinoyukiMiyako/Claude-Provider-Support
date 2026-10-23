"""MaiBot 统一请求快照与 Claude Messages API 之间的映射。

只做数据转换，不发网络请求；网络调用和客户端生命周期在 plugin.py。
"""

from __future__ import annotations

import hashlib
import re
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
# 只有这些模型在官方 Claude API 上接受 fallbacks="default"；对其他模型发送该参数会被拒绝或不起作用
SERVER_FALLBACK_MODELS = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"}
)
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
_TRUNCATION_ADVICE = {
    "max_tokens": (" max_tokens 上限", "请调大该任务或模型的 max_tokens"),
    "model_context_window_exceeded": ("模型的上下文窗口上限", "请缩短该任务的聊天历史或注入内容"),
}
_TOOL_ID = re.compile(r"[A-Za-z0-9_-]+")
_TOOL_ID_INVALID_CHAR = re.compile(r"[^A-Za-z0-9_-]")


class ClaudeProviderError(RuntimeError):
    """请求无法映射为 Messages API，或响应无法交还给 MaiBot。"""


@dataclass(frozen=True, slots=True)
class ModelOptions:
    effort: str | None
    thinking: str
    thinking_display: str | None


def build_message_params(
    request: Mapping[str, Any], settings: ProviderSettings, *, official_endpoint: bool
) -> dict[str, Any]:
    """把 MaiBot 的 response 请求快照转换为 ``client.beta.messages.stream`` 的关键字参数。

    ``official_endpoint`` 表示请求发往官方 Claude API；兼容该协议的第三方端点不认识服务端拒答兜底。
    """

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
        # 不用请求级自动断点：MaiBot 每轮末尾都拼着只属于这一轮的内容，自动断点落在那里只写不读，
        # 白付 1.25 倍（1h 为 2 倍）的写入价
        _mark_shared_prefix_end(messages, cache_control)
    temperature = request.get("temperature")
    if temperature is not None:
        # anthropic SDK 已去掉 temperature 形参，官方 API 自 2026-10-21 版起对任何模型都返回 400；
        # 兼容端点多数仍接受，开着 send_temperature 就经 extra_body 原样发出，由端点决定
        params["extra_body"] = {"temperature": temperature}
    if settings.refusal_fallback and official_endpoint and model in SERVER_FALLBACK_MODELS:
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
                # 对话中途的 system 消息先原位放置，位置不合法的由 _demote_misplaced_system 统一改写
                _append_message(messages, "system", blocks)
            else:
                system_blocks.extend(blocks)
        elif item_type == "UserMessageItem":
            _append_user_blocks(messages, _user_blocks(item.get("parts"), where))
        elif item_type == "AssistantMessageItem":
            _append_message(messages, "assistant", _assistant_blocks(item.get("parts"), where))
        elif item_type == "FunctionCallItem":
            _append_message(messages, "assistant", [_tool_use_block(item.get("tool_call"), where)])
        elif item_type == "FunctionCallOutputItem":
            _append_tool_result(messages, _tool_result_block(item, where))
        elif item_type in _SKIPPED_ITEM_TYPES:
            continue
        else:
            raise ClaudeProviderError(f"{where} 的 item_type 无法转换: {item_type!r}")
    _demote_misplaced_system(messages)
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
        text = "".join(pending_text)
        pending_text.clear()
        # 纯空白正文交回 MaiBot 后会作为 assistant 历史再发出来，整条只有空白的消息会被 API 拒绝
        if text.strip():
            output_items.append({"item_type": "AssistantMessageItem", "parts": [{"type": "text", "text": text}]})

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

    if stop_reason in _TRUNCATION_ADVICE:
        # 截断的响应已计费，原样重试只会再付一次输入费用得到同样的结果；有正文就交回已生成的部分
        limit, advice = _TRUNCATION_ADVICE[stop_reason]
        if any(item["item_type"] == "FunctionCallItem" for item in output_items):
            raise ClaudeProviderError(f"输出在工具调用中途达到{limit}，参数不完整，{advice}")
        if not any(item["item_type"] == "AssistantMessageItem" for item in output_items):
            raise ClaudeProviderError(f"产出正文前就达到{limit}，{advice}")
    elif stop_reason not in _NORMAL_STOP_REASONS:
        raise ClaudeProviderError(f"未处理的 stop_reason: {stop_reason!r}")
    elif not output_items:
        # 与 MaiBot 内置客户端的空响应处理一致：报错让 Host 按 max_retry 重试，而不是交回一个空回复
        raise ClaudeProviderError(f"响应没有正文、工具调用或思考摘要（stop_reason={stop_reason}）")

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


def _mark_shared_prefix_end(messages: list[dict[str, Any]], cache_control: Mapping[str, Any]) -> None:
    # MaiBot 每轮都在末尾重新拼接时间、人物画像、回复方向等一次性内容，断点落在这段之后下一轮就读不到。
    # 最后一块模型产出（回复、工具调用或其结果）之前的内容下一轮必然原样出现，断点打在这里，
    # 下一轮从自己的断点回看 20 个位置即可命中整段历史
    for message in reversed(messages):
        content = message["content"]
        for block_index in range(len(content) - 1, -1, -1):
            block = content[block_index]
            if message["role"] == "assistant" or block.get("type") == "tool_result":
                content[block_index] = {**block, "cache_control": dict(cache_control)}
                return


def _append_user_blocks(messages: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> None:
    # 缓存断点只回看 20 个块。群友连发一串消息时每条都单独成块，下一轮的历史断点会够不着上一轮写入的位置，
    # 所以相邻用户消息的文字用换行接成一个块；图片和工具结果仍各自成块
    if messages and messages[-1]["role"] == "user" and blocks:
        content = messages[-1]["content"]
        if content and content[-1]["type"] == "text" and blocks[0]["type"] == "text":
            content[-1] = {"type": "text", "text": f"{content[-1]['text']}\n{blocks[0]['text']}"}
            blocks = blocks[1:]
    _append_message(messages, "user", blocks)


def _append_tool_result(messages: list[dict[str, Any]], block: dict[str, Any]) -> None:
    # API 只在紧接 tool_use 的 user 消息开头找 tool_result；第三方插件可能先给文字再给结果，
    # 原样追加会排在文字后面而被判为没有对应结果
    if messages and messages[-1]["role"] == "user":
        content = messages[-1]["content"]
        position = next((index for index, existing in enumerate(content) if existing["type"] != "tool_result"), None)
        content.insert(len(content) if position is None else position, block)
    else:
        messages.append({"role": "user", "content": [block]})


def _demote_misplaced_system(messages: list[dict[str, Any]]) -> None:
    # API 只接受紧跟 user、且位于末尾或紧接 assistant 的中途 system，其他位置一律 400；
    # 这些位置的 system 改作 user 文本，内容仍送达模型
    placed: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        following = messages[index + 1]["role"] if index + 1 < len(messages) else None
        if message["role"] == "system":
            if placed and placed[-1]["role"] == "user" and following in (None, "assistant"):
                placed.append(message)
            else:
                _merge_user_content(placed, message["content"])
        elif message["role"] == "user" and placed and placed[-1]["role"] == "user":
            _merge_user_content(placed, message["content"])
        else:
            placed.append(message)
    messages[:] = placed


def _merge_user_content(messages: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> None:
    for block in blocks:
        if block["type"] == "tool_result":
            _append_tool_result(messages, block)
    _append_user_blocks(messages, [block for block in blocks if block["type"] != "tool_result"])


def _append_message(messages: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    if not blocks:
        # 片段全被滤掉（例如只有空白）时不产生空消息，content 为空的消息会被 API 拒绝
        return
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
            text = _require_text(part.get("text"), f"{where}.text")
            if blocks and blocks[-1]["type"] == "text":
                # MaiBot 会把一条消息按表情等切成多段（一句歌词能切出几十段），原样拼回同一条消息
                blocks[-1] = {"type": "text", "text": blocks[-1]["text"] + text}
            else:
                blocks.append({"type": "text", "text": text})
        elif part_type == "image":
            blocks.append(_image_block(part, where))
        else:
            raise ClaudeProviderError(f"{where} 的用户消息片段类型无法转换: {part_type!r}")
    return _drop_blank_text(blocks)


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
    return _drop_blank_text(blocks)


def _drop_blank_text(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Host 只拦空串不拦纯空白；整条消息只剩空白文本时 API 返回 400，空白块本身也不携带信息
    return [block for block in blocks if block["type"] != "text" or block["text"].strip()]


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
        "id": _tool_id(tool_call.get("call_id"), f"{where}.tool_call.call_id"),
        "name": _require_text(tool_call.get("func_name"), f"{where}.tool_call.func_name"),
        "input": dict(_require_mapping(tool_call.get("args"), f"{where}.tool_call.args")),
    }


def _tool_result_block(item: Mapping[str, Any], where: str) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": _tool_id(item.get("call_id"), f"{where}.call_id"),
    }
    output = item.get("output")
    if not isinstance(output, str):
        raise ClaudeProviderError(f"{where}.output 必须是字符串，实际为 {type(output).__name__}")
    if output:
        block["content"] = output
    if item.get("success") is False:
        block["is_error"] = True
    return block


def _tool_id(raw_id: Any, where: str) -> str:
    call_id = _require_text(raw_id, where)
    if _TOOL_ID.fullmatch(call_id):
        return call_id
    # 任务的 model_list 混用多家 Provider 时，历史里会留着别家生成的 call_id，API 只接受 [A-Za-z0-9_-]；
    # 替换字符后追加原值摘要，不同 id 不会撞成同一个，同一 id 每轮结果相同也不破坏缓存前缀
    digest = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:12]
    return f"{_TOOL_ID_INVALID_CHAR.sub('_', call_id)}_{digest}"


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
