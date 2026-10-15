"""用真实 Claude API 冒烟：参数组合被接受、同前缀第二次请求命中缓存、工具调用不回放思考也能续接。

只走插件的 handle_llm_request，不依赖 MaiBot。共发 4 次小请求，用 Opus 5.5 时总花费约 0.1 美元以内，Haiku 5.5 更低。

    $env:ANTHROPIC_API_KEY = "<你的 Key>"
    $env:ANTHROPIC_WORKSPACE_ID = "wrkspc_..."   # 仅当 Key 未限定到单个工作区时需要
    uv run python scripts/live_smoke.py --model claude-haiku-5-5
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

PLUGIN_DIR = Path(__file__).resolve().parents[1]
# 一段稳定且超过 Opus 5.5 最小缓存长度(512 token)的 system 提示
SYSTEM_PROMPT = "你是群聊机器人麦麦，回复简短口语化。\n" + "\n".join(
    f"守则 {index}：遇到第 {index} 类话题时先确认对方意图，再给出不超过两句的回答，不编造事实。"
    for index in range(1, 61)
)
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市今天的天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def load_plugin() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "claude_provider_live", str(PLUGIN_DIR / "plugin.py"), submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["claude_provider_live"] = module
    spec.loader.exec_module(module)
    return module


def item(item_type: str, text: str, turn: str | None = None) -> dict[str, Any]:
    return {
        "item_type": item_type,
        "meta": {"item_id": f"{item_type}-{text[:6]}", "logical_turn_id": turn, "timestamp": "2026-10-15T12:00:00"},
        "parts": [{"type": "text", "text": text}],
    }


def build_request(model: str, provider: dict[str, Any], items: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "item_schema_version": 1,
        "extra_params": {},
        "max_tokens": 4096,
        "context_items": items,
        "logical_turn_id": "live",
        "model_info": {"model_identifier": model, "name": model, "extra_params": {}},
        "request_kind": "response",
        "response_format": None,
        "temperature": None,
        "tool_options": [],
        "api_provider": provider,
        **extra,
    }


def text_of(result: dict[str, Any]) -> str:
    return "".join(
        part["text"]
        for output in result["output_items"]
        if output["item_type"] == "AssistantMessageItem"
        for part in output["parts"]
    )


async def run(model: str) -> int:
    plugin_module = load_plugin()
    plugin = plugin_module.create_plugin()
    plugin.set_plugin_config({})
    provider = {
        "base_url": "https://api.anthropic.com",
        "api_key": os.environ["ANTHROPIC_API_KEY"],
        "auth_type": "bearer",
        # 未限定工作区的个人/服务账号 Key 必须带这个头，否则 API 返回 400
        "default_headers": (
            {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]}
            if os.environ.get("ANTHROPIC_WORKSPACE_ID")
            else {}
        ),
        "default_query": {},
        "timeout": 180,
    }
    checks: list[tuple[str, bool, str]] = []
    try:
        system = item("SystemMessageItem", SYSTEM_PROMPT)
        first_user = item("UserMessageItem", "用一句话介绍你自己")
        first = await plugin.handle_llm_request(
            operation="response", request=build_request(model, provider, [system, first_user])
        )
        checks.append(("首个请求被接受并返回正文", bool(text_of(first)), text_of(first)))

        second = await plugin.handle_llm_request(
            operation="response",
            request=build_request(
                model,
                provider,
                [system, first_user, item("AssistantMessageItem", text_of(first)), item("UserMessageItem", "再短一点")],
            ),
        )
        hit = second["usage"].get("prompt_cache_hit_tokens", 0)
        checks.append(("同前缀第二次请求命中缓存", hit > 0, json.dumps(second["usage"], ensure_ascii=False)))

        weather_user = item("UserMessageItem", "上海今天天气怎么样？请先调用 get_weather 查询")
        tool_turn = await plugin.handle_llm_request(
            operation="response",
            request=build_request(model, provider, [system, weather_user], tool_options=[WEATHER_TOOL]),
        )
        calls = [output for output in tool_turn["output_items"] if output["item_type"] == "FunctionCallItem"]
        checks.append(("模型发起工具调用", bool(calls), json.dumps(calls, ensure_ascii=False)))
        if calls:
            call = calls[0]["tool_call"]
            replay = [system, weather_user]
            if text_of(tool_turn):
                replay.append(item("AssistantMessageItem", text_of(tool_turn), "tool-turn"))
            replay.append(
                {
                    "item_type": "FunctionCallItem",
                    "meta": {**system["meta"], "item_id": "call", "logical_turn_id": "tool-turn"},
                    "tool_call": {**call, "extra_content": None},
                }
            )
            replay.append(
                {
                    "item_type": "FunctionCallOutputItem",
                    "meta": {**system["meta"], "item_id": "out", "logical_turn_id": "tool-turn"},
                    "call_id": call["call_id"],
                    "output": "晴，22 度",
                    "success": True,
                    "tool_name": call["func_name"],
                }
            )
            final = await plugin.handle_llm_request(
                operation="response",
                request=build_request(model, provider, replay, tool_options=[WEATHER_TOOL]),
            )
            checks.append(("不回放思考块也能续接工具结果", "22" in text_of(final), text_of(final)))
    finally:
        await plugin.on_unload()

    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-haiku-5-5")
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("未设置 ANTHROPIC_API_KEY 环境变量")
    raise SystemExit(asyncio.run(run(args.model)))


if __name__ == "__main__":
    main()
