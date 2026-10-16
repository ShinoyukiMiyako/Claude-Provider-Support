from __future__ import annotations

from typing import Literal

from maibot_sdk import Field, PluginConfigBase


class ProviderSettings(PluginConfigBase):
    __ui_label__ = "Claude Provider Support 设置"
    __ui_icon__ = "brain"

    config_version: str = Field(
        default="1.0.0",
        description="插件配置结构版本",
        json_schema_extra={"label": {"zh_CN": "配置版本"}},
    )
    prompt_cache: bool = Field(
        default=True,
        description=(
            "开启后对 system 与整段对话打缓存断点；命中部分按缓存读取价计费，写入部分按 1.25 倍(5m)或 2 倍(1h)计费"
        ),
        json_schema_extra={"x-widget": "switch", "label": {"zh_CN": "提示缓存"}},
    )
    cache_ttl: Literal["5m", "1h"] = Field(
        default="5m",
        description="缓存存活时间；群聊消息间隔经常超过 5 分钟时改用 1h",
        json_schema_extra={"x-widget": "select", "label": {"zh_CN": "缓存时长"}},
    )
    thinking_display: Literal["summarized", "omitted"] = Field(
        default="summarized",
        description="summarized 返回思考摘要并显示为 reasoning；omitted 不返回思考文本，计费不变",
        json_schema_extra={"x-widget": "select", "label": {"zh_CN": "思考内容"}},
    )
    refusal_fallback: bool = Field(
        default=True,
        description="模型因安全策略拒答时，由 Anthropic 服务端按拒答类别自动换模型重答；只对支持该能力的模型生效",
        json_schema_extra={"x-widget": "switch", "label": {"zh_CN": "拒答兜底"}},
    )


class ClaudeProviderConfig(PluginConfigBase):
    plugin: ProviderSettings = Field(default_factory=ProviderSettings)
