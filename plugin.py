from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic
from maibot_sdk import LLMProvider, MaiBotPlugin

from .claude_adapter import (
    CLIENT_TYPE,
    ClaudeProviderError,
    build_message_params,
    parse_message,
)
from .config_models import ClaudeProviderConfig, ProviderSettings

PLUGIN_VERSION = "0.1.0"
WORKSPACE_HEADER = "anthropic-workspace-id"
_WORKSPACE_ID = re.compile(r"wrkspc_\S+")


class ClaudeProviderPlugin(MaiBotPlugin):
    config_model = ClaudeProviderConfig

    def __init__(self) -> None:
        super().__init__()
        self._clients: dict[tuple[str, ...], AsyncAnthropic] = {}

    async def on_load(self) -> None:
        self._get_logger().info("Claude Provider 已加载，client_type=%s", CLIENT_TYPE)

    async def on_unload(self) -> None:
        clients = list(self._clients.values())
        self._clients.clear()
        for client in clients:
            await client.close()

    async def on_config_update(self, scope: str, config_data: dict[str, Any], version: str) -> None:
        # Runner 在回调前已把 self 范围的新配置注入，这里每次请求都现读 self.config，无需额外处理
        self._get_logger().info("Claude Provider 配置已更新 scope=%s version=%s", scope, version)

    @LLMProvider(
        CLIENT_TYPE,
        name="Claude Messages API",
        description="通过官方 anthropic SDK 调用 Claude Messages API。",
        version=PLUGIN_VERSION,
    )
    async def handle_llm_request(self, *, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        if operation != "response":
            raise ClaudeProviderError(f"Claude API 不提供 {operation} 能力，请为该任务改用其他 Provider")

        provider = request.get("api_provider")
        if not isinstance(provider, Mapping):
            raise ClaudeProviderError("请求缺少 api_provider 配置快照")
        params = build_message_params(request, self._settings)
        client = self._get_client(provider)
        timeout = float(provider["timeout"])
        try:
            # Host 等待 RPC 的时长就是 Provider 的 timeout；流式读超时按字节间隔计，
            # 不限制总时长，所以在这里兜住总时长，避免 Host 放弃后请求还在后台跑
            async with asyncio.timeout(timeout):
                async with client.beta.messages.stream(**params) as stream:
                    message = await stream.get_final_message()
        except TimeoutError as exc:
            raise ClaudeProviderError(
                f"Claude 请求超过 {timeout:g}s 仍未完成，已中止；可调大该 Provider 的 timeout"
            ) from exc

        result = parse_message(message)
        logger = self._get_logger()
        raw_data = result["raw_data"]
        if raw_data["served_by_fallback"]:
            logger.warning("模型 %s 拒答，本轮已由 %s 兜底回答", params["model"], raw_data["model"])
        if raw_data["stop_reason"] == "max_tokens":
            logger.warning(
                "模型 %s 的输出达到 max_tokens=%s 上限，正文可能不完整", params["model"], params["max_tokens"]
            )
        return result

    @property
    def _settings(self) -> ProviderSettings:
        return cast(ClaudeProviderConfig, self.config).plugin

    def _get_client(self, provider: Mapping[str, Any]) -> AsyncAnthropic:
        auth_type = provider.get("auth_type")
        if auth_type != "bearer":
            raise ClaudeProviderError(
                f"Claude 官方 API 固定使用 x-api-key 鉴权，auth_type 请保持默认值 bearer，实际为 {auth_type!r}"
            )
        api_key = provider.get("api_key")
        if not isinstance(api_key, str) or not api_key:
            raise ClaudeProviderError("api_provider.api_key 为空，请在模型配置的 Provider 里填写 Claude API Key")
        base_url = _validate_base_url(provider.get("base_url"))
        headers = _validate_headers(provider.get("default_headers"))
        query = dict(provider.get("default_query") or {})
        timeout = float(provider["timeout"])

        signature = (
            api_key,
            base_url,
            json.dumps(headers, sort_keys=True, ensure_ascii=False),
            json.dumps(query, sort_keys=True, ensure_ascii=False),
            str(timeout),
        )
        client = self._clients.get(signature)
        if client is None:
            client = AsyncAnthropic(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                # Host 会按 Provider 的 max_retry 重试插件失败的请求，SDK 再重试会让总耗时超出 Host 的等待
                max_retries=0,
                default_headers=headers or None,
                default_query=query or None,
            )
            self._clients[signature] = client
        return client


def _validate_base_url(raw_base_url: Any) -> str:
    if not isinstance(raw_base_url, str) or not raw_base_url.strip():
        raise ClaudeProviderError("api_provider.base_url 为空，官方地址为 https://api.anthropic.com")
    base_url = raw_base_url.strip().rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ClaudeProviderError(f"api_provider.base_url 不是有效的 http(s) 地址: {raw_base_url!r}")
    if parsed.path.endswith("/v1"):
        # SDK 会自己拼上 /v1/messages，带着 /v1 会请求到 /v1/v1/messages
        raise ClaudeProviderError(f"api_provider.base_url 不要带 /v1，请改为 {base_url[: -len('/v1')]}")
    return base_url


def _validate_headers(raw_headers: Any) -> dict[str, Any]:
    headers = dict(raw_headers or {})
    for name, value in headers.items():
        # 未限定工作区的 Key（个人 Key、多工作区服务账号 Key）靠这个头选工作区；
        # 复制时带进空格或填成工作区名称，API 只会回一个看不出原因的 400
        if name.lower() == WORKSPACE_HEADER and not (isinstance(value, str) and _WORKSPACE_ID.fullmatch(value)):
            raise ClaudeProviderError(
                f"default_headers 中的 {name} 应为 Console Workspaces 页 ID 列的 wrkspc_... 值，"
                f"不含空格和引号，实际为 {value!r}"
            )
    return headers


def create_plugin() -> ClaudeProviderPlugin:
    return ClaudeProviderPlugin()
