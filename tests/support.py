from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "claude_provider_under_test"


def load_plugin_package() -> ModuleType:
    """按 MaiBot plugin_loader 的方式导入插件：plugin.py 作包入口，插件目录作子模块搜索路径。"""

    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        str(PLUGIN_DIR / "plugin.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


class FakeStream:
    """替代 ``client.beta.messages.stream(...)`` 返回的异步上下文管理器。"""

    def __init__(self, message: Any, delay: float = 0.0) -> None:
        self.message = message
        self.delay = delay
        self.exited = False

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        self.exited = True
        return False

    async def get_final_message(self) -> Any:
        await asyncio.sleep(self.delay)
        return self.message


class FakeClient:
    """只记录 stream 调用参数的 AsyncAnthropic 替身，不发网络请求。"""

    def __init__(self, stream: FakeStream) -> None:
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self._stream_obj = stream

    def _stream(self, **params: Any) -> FakeStream:
        self.calls.append(params)
        return self._stream_obj


def make_message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
    model: str = "claude-opus-5-5",
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    """用 SDK 自己的响应模型构造消息，保证字段名与真实响应一致。"""

    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage
            or {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
    )


def meta(turn: str | None = None) -> dict[str, Any]:
    return {"item_id": "item", "logical_turn_id": turn, "timestamp": "2026-10-15T12:00:00"}


def text_item(item_type: str, text: str) -> dict[str, Any]:
    return {"item_type": item_type, "meta": meta(), "parts": [{"type": "text", "text": text}]}


def make_request(context_items: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "item_schema_version": 1,
        "extra_params": {},
        "max_tokens": 4096,
        "context_items": context_items,
        "logical_turn_id": "turn-1",
        "model_info": {"model_identifier": "claude-opus-5-5", "name": "opus", "extra_params": {}},
        "request_kind": "response",
        "response_format": None,
        "temperature": None,
        "tool_options": [],
    }
    request.update(overrides)
    return request
