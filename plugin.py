from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic, omit
from maibot_sdk import LLMProvider, MaiBotPlugin

from .claude_adapter import (
    CLIENT_TYPE,
    ClaudeProviderError,
    build_message_params,
    parse_message,
)
from .config_models import ClaudeProviderConfig, ProviderSettings

PLUGIN_VERSION = "2.0.0"
OFFICIAL_API_HOST = "api.anthropic.com"
AUTH_TYPES = ("bearer", "header", "query", "none")
WORKSPACE_HEADER = "anthropic-workspace-id"
BETA_HEADER = "anthropic-beta"
HOST_RPC_MARGIN_SECONDS = 1.0
_WORKSPACE_ID = re.compile(r"wrkspc_[A-Za-z0-9]+")
# 决定客户端行为的 Provider 字段；任一变化都要换一个客户端
_CLIENT_FIELDS = (
    "api_key",
    "base_url",
    "auth_type",
    "auth_header_name",
    "auth_header_prefix",
    "auth_query_name",
    "default_headers",
    "default_query",
    "timeout",
)


class ClaudeProviderPlugin(MaiBotPlugin):
    config_model = ClaudeProviderConfig

    def __init__(self) -> None:
        super().__init__()
        self._clients: dict[str, AsyncAnthropic] = {}

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
        description="通过官方 anthropic SDK 调用 Claude Messages API 或兼容该协议的端点。",
        version=PLUGIN_VERSION,
    )
    async def handle_llm_request(self, *, operation: str, request: dict[str, Any]) -> dict[str, Any]:
        if operation != "response":
            raise ClaudeProviderError(f"Claude API 不提供 {operation} 能力，请为该任务改用其他 Provider")

        provider = request.get("api_provider")
        if not isinstance(provider, Mapping):
            raise ClaudeProviderError("请求缺少 api_provider 配置快照")
        official = urlsplit(_validate_base_url(provider.get("base_url"))).hostname == OFFICIAL_API_HOST
        params = build_message_params(request, self._settings, official_endpoint=official)
        if "betas" in params:
            params["betas"] = _merge_configured_betas(provider.get("default_headers"), params["betas"])
        client = self._get_client(provider)
        timeout = float(provider["timeout"])
        try:
            # 流式读超时按字节间隔计，不限制总时长，所以在这里兜住总时长，避免 Host 放弃后请求还在后台跑。
            # Host 等待 RPC 的时长同样是 timeout 且更早开始计时，留出余量才能让下面的超时说明先送到
            async with asyncio.timeout(max(timeout - HOST_RPC_MARGIN_SECONDS, timeout / 2)):
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
        elif raw_data["stop_reason"] == "model_context_window_exceeded":
            logger.warning("模型 %s 的输入加输出达到上下文窗口上限，正文可能不完整", params["model"])
        return result

    @property
    def _settings(self) -> ProviderSettings:
        return cast(ClaudeProviderConfig, self.config).plugin

    def _get_client(self, provider: Mapping[str, Any]) -> AsyncAnthropic:
        base_url = _validate_base_url(provider.get("base_url"))
        api_key, auth_headers, auth_query = _resolve_auth(provider)
        # 与 MaiBot 内置客户端一致：鉴权生成的头和查询参数覆盖 Provider 里同名的配置
        headers = {**_validate_headers(provider.get("default_headers")), **auth_headers}
        query = {**dict(provider.get("default_query") or {}), **auth_query}
        timeout = float(provider["timeout"])

        signature = json.dumps(
            {field: provider.get(field) for field in _CLIENT_FIELDS}, sort_keys=True, ensure_ascii=False
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


def _resolve_auth(provider: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, str]]:
    """按 Provider 的 auth_type 返回 (交给 SDK 的 api_key, 鉴权请求头, 鉴权查询参数)。

    语义与 MaiBot 内置 OpenAI 兼容客户端相同，同一个 Provider 改成 claude 客户端后鉴权方式不变。
    """

    auth_type = provider.get("auth_type")
    if auth_type not in AUTH_TYPES:
        raise ClaudeProviderError(f"auth_type 只能是 {' / '.join(AUTH_TYPES)} 之一，实际为 {auth_type!r}")
    # 除默认的 Authorization: Bearer 外都不让 SDK 生成凭据头。SDK 没拿到显式凭据时会去读 ANTHROPIC_API_KEY
    # 等环境变量和本机 profile，会把宿主上的 Key 发给第三方端点，所以传空串占位，再去掉它据此生成的头
    if auth_type == "none":
        return "", {"Authorization": omit}, {}
    api_key = provider.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        raise ClaudeProviderError(
            "api_provider.api_key 为空，请在模型配置的 Provider 里填写 API Key；端点不需要鉴权时把 auth_type 设为 none"
        )
    if auth_type == "query":
        return "", {"Authorization": omit}, {_require_provider_text(provider, "auth_query_name"): api_key}

    header_name = _require_provider_text(provider, "auth_header_name")
    prefix = provider.get("auth_header_prefix")
    if not isinstance(prefix, str):
        raise ClaudeProviderError(f"api_provider.auth_header_prefix 必须是字符串，实际为 {prefix!r}")
    if auth_type == "bearer" and header_name == "Authorization" and prefix.strip() == "Bearer":
        return api_key, {}, {}
    value = f"{prefix.strip()} {api_key}" if prefix.strip() else api_key
    if header_name.lower() == "authorization":
        return "", {header_name: value}, {}
    return "", {"Authorization": omit, header_name: value}, {}


def _require_provider_text(provider: Mapping[str, Any], key: str) -> str:
    value = provider.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ClaudeProviderError(f"auth_type={provider.get('auth_type')!r} 时 api_provider.{key} 不能为空")
    return value.strip()


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


def _merge_configured_betas(raw_headers: Any, betas: list[str]) -> list[str]:
    # SDK 由请求级 betas 生成的 anthropic-beta 头会整体覆盖 default_headers 里的同名头，
    # 不合并的话 Provider 里配的 beta 在启用兜底的模型上会静默失效
    configured = [
        beta.strip()
        for name, value in dict(raw_headers or {}).items()
        if name.lower() == BETA_HEADER
        for beta in str(value).split(",")
    ]
    return list(dict.fromkeys([*filter(None, configured), *betas]))


def create_plugin() -> ClaudeProviderPlugin:
    return ClaudeProviderPlugin()
