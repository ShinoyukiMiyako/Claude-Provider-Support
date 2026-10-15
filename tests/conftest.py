from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest
from support import PACKAGE_NAME, load_plugin_package


@pytest.fixture(scope="session")
def plugin_module() -> ModuleType:
    return load_plugin_package()


@pytest.fixture(scope="session")
def adapter(plugin_module: ModuleType) -> ModuleType:
    return importlib.import_module(f"{PACKAGE_NAME}.claude_adapter")


@pytest.fixture(scope="session")
def config_models(plugin_module: ModuleType) -> ModuleType:
    return importlib.import_module(f"{PACKAGE_NAME}.config_models")


@pytest.fixture
def settings(config_models: ModuleType) -> Any:
    return config_models.ProviderSettings()
