"""与 MaiBot 宿主的契约测试：用宿主自己的类型构造请求、加载插件、回收输出。

需要 MaiBot 的完整依赖，所以只在 MaiBot 的虚拟环境里、并设置 MAIBOT_ROOT 指向 MaiBot 仓库时运行：

    $env:MAIBOT_ROOT = "D:\\Repo\\MaiBot"
    D:\\Repo\\MaiBot\\.venv\\Scripts\\python.exe -m pytest tests/test_maibot_contract.py
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from support import PLUGIN_DIR, FakeClient, FakeStream, make_message

MAIBOT_ROOT = os.environ.get("MAIBOT_ROOT", "")
pytestmark = pytest.mark.skipif(
    not MAIBOT_ROOT,
    reason="未设置 MAIBOT_ROOT；宿主契约测试需要指向 MaiBot 仓库并在其虚拟环境中运行",
)
PLUGIN_FILES = ("_manifest.json", "plugin.py", "claude_adapter.py", "config_models.py")


class Score(BaseModel):
    """回复评分"""

    score: int
    reason: str


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    # 宿主模块按工作目录解析 src/ 与配置文件，必须在 MaiBot 根目录下导入
    monkeypatch.chdir(MAIBOT_ROOT)
    if MAIBOT_ROOT not in sys.path:
        sys.path.insert(0, MAIBOT_ROOT)
    from src.config import model_configs
    from src.llm_models.model_client import base_client, plugin_client
    from src.llm_models.payload_content import context_item, resp_format, tool_option
    from src.plugin_runtime.runner import manifest_validator, plugin_loader

    return SimpleNamespace(
        APIProvider=model_configs.APIProvider,
        ModelInfo=model_configs.ModelInfo,
        ResponseRequest=base_client.ResponseRequest,
        PluginLLMClient=plugin_client.PluginLLMClient,
        AssistantMessageItem=context_item.AssistantMessageItem,
        ContextItemBuilder=context_item.ContextItemBuilder,
        ContextItemMeta=context_item.ContextItemMeta,
        ContextToolCall=context_item.ContextToolCall,
        FunctionCallItem=context_item.FunctionCallItem,
        FunctionCallOutputItem=context_item.FunctionCallOutputItem,
        ReasoningItem=context_item.ReasoningItem,
        ReasoningRepresentation=context_item.ReasoningRepresentation,
        RoleType=context_item.RoleType,
        RespFormat=resp_format.RespFormat,
        RespFormatType=resp_format.RespFormatType,
        ToolOption=tool_option.ToolOption,
        # 直接导入 request_snapshot 会撞上宿主内部的循环导入，借用 plugin_client 已解析好的同一函数
        serialize_response_request_snapshot=plugin_client.serialize_response_request_snapshot,
        ManifestValidator=manifest_validator.ManifestValidator,
        PluginLoader=plugin_loader.PluginLoader,
    )


def _build_request(host: SimpleNamespace) -> Any:
    image_base64 = base64.b64encode(b"\xff\xd8\xff fake jpeg").decode("ascii")
    call = host.FunctionCallItem(
        meta=host.ContextItemMeta.create(logical_turn_id="turn-1"),
        tool_call=host.ContextToolCall.create(call_id="call_1", func_name="search", args={"q": "猫"}),
    )
    output = host.FunctionCallOutputItem(
        meta=host.ContextItemMeta.create(logical_turn_id="turn-1"),
        call_id="call_1",
        output="找到 3 条",
        tool_name="search",
    )
    model_info = host.ModelInfo(
        model_type="llm",
        name="opus",
        model_identifier="claude-opus-5-5",
        api_provider="Claude",
        send_temperature=False,
        visual=True,
        extra_params={"effort": "high"},
    )
    return host.ResponseRequest(
        model_info=model_info,
        context_items=[
            host.ContextItemBuilder().set_role(host.RoleType.System).add_text_content("你是麦麦").build(),
            host.ContextItemBuilder()
            .add_text_content("看图")
            .add_image_content(image_format="jpg", image_base64=image_base64)
            .build(),
            call,
            output,
            host.ContextItemBuilder().add_text_content("总结一下").build(),
        ],
        tool_options=[
            host.ToolOption.from_definition(
                {
                    "name": "search",
                    "description": "搜索",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                }
            )
        ],
        max_tokens=2048,
        response_format=host.RespFormat(format_type=host.RespFormatType.JSON_SCHEMA, schema=Score),
        extra_params=dict(model_info.extra_params),
    )


def test_manifest_passes_host_validator_with_dependency_check(host: SimpleNamespace) -> None:
    validator = host.ManifestValidator()

    manifest = validator.parse_manifest(
        json.loads((PLUGIN_DIR / "_manifest.json").read_text(encoding="utf-8")),
        source=str(PLUGIN_DIR / "_manifest.json"),
    )

    assert manifest is not None, validator.errors
    assert manifest.llm_provider_client_types == ["claude"]


def test_host_loader_loads_plugin_and_registers_provider(host: SimpleNamespace, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "claude_provider"
    plugin_dir.mkdir(parents=True)
    for name in PLUGIN_FILES:
        shutil.copy2(PLUGIN_DIR / name, plugin_dir / name)
    loader = host.PluginLoader()

    metas = loader.discover_and_load([str(plugin_dir.parent)])

    assert loader.failed_plugins == {}
    assert [meta.plugin_id for meta in metas] == ["shinoyukimiyako.claude-provider"]
    providers = metas[0].instance.get_llm_providers()
    assert [(provider["client_type"], provider["metadata"]["handler_name"]) for provider in providers] == [
        ("claude", "handle_llm_request")
    ]


def test_host_snapshot_maps_to_claude_params(host: SimpleNamespace, adapter: Any, settings: Any) -> None:
    snapshot = host.serialize_response_request_snapshot(_build_request(host))

    params = adapter.build_message_params(snapshot, settings)

    assert params["system"][0]["text"] == "你是麦麦"
    assert [message["role"] for message in params["messages"]] == ["user", "assistant", "user"]
    assert params["messages"][0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert params["messages"][1]["content"] == [
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "猫"}}
    ]
    assert params["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "找到 3 条",
    }
    assert params["tools"][0]["input_schema"]["required"] == ["q"]
    assert params["output_config"]["effort"] == "high"
    schema = params["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["reason", "score"]
    assert "extra_body" not in params


async def test_full_round_trip_through_plugin_llm_client(
    host: SimpleNamespace, plugin_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_reply = make_message(
        [
            {"type": "thinking", "thinking": "用户要总结", "signature": "sig"},
            {"type": "text", "text": '{"score": 8, "reason": "清楚"}'},
        ],
        usage={
            "input_tokens": 40,
            "output_tokens": 12,
            "cache_creation_input_tokens": 60,
            "cache_read_input_tokens": 500,
        },
    )
    fake_client = FakeClient(FakeStream(claude_reply))
    constructed: list[dict[str, Any]] = []

    def fake_async_anthropic(**kwargs: Any) -> FakeClient:
        constructed.append(kwargs)
        return fake_client

    monkeypatch.setattr(plugin_module, "AsyncAnthropic", fake_async_anthropic)
    plugin = plugin_module.create_plugin()
    plugin.set_plugin_config({})

    class Supervisor:
        async def invoke_llm_provider(self, *, plugin_id, client_type, operation, request, timeout_ms):
            result = await plugin.handle_llm_request(operation=operation, request=request)
            return SimpleNamespace(error=None, payload={"success": True, "result": result})

    api_provider = host.APIProvider(
        name="Claude",
        base_url="https://api.anthropic.com",
        api_key="sk-ant-test-placeholder",
        client_type="claude",
        timeout=300,
    )
    client = host.PluginLLMClient(api_provider, Supervisor(), "shinoyukimiyako.claude-provider", "claude")

    response = await client.get_response(_build_request(host))

    assert constructed[0]["base_url"] == "https://api.anthropic.com"
    assert constructed[0]["api_key"] == "sk-ant-test-placeholder"
    assert constructed[0]["timeout"] == 300.0
    assert fake_client.calls[0]["model"] == "claude-opus-5-5"
    reasoning, assistant = response.output_items
    assert isinstance(reasoning, host.ReasoningItem)
    assert reasoning.representation == host.ReasoningRepresentation.SUMMARY
    assert reasoning.summary_parts == ("用户要总结",)
    assert isinstance(assistant, host.AssistantMessageItem)
    assert assistant.parts[0].text == '{"score": 8, "reason": "清楚"}'
    assert response.usage.prompt_tokens == 600
    assert response.usage.prompt_cache_hit_tokens == 500
    assert response.usage.prompt_cache_miss_tokens == 100
    assert response.usage.prompt_cache_reported is True
