"""校验插件结构（需要安装 maibot-plugin-sdk，未安装时自动跳过）。"""

import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

maibot_sdk = pytest.importorskip("maibot_sdk")


def test_create_plugin_returns_instance():
    import plugin

    instance = plugin.create_plugin()
    assert isinstance(instance, plugin.JmcomicPlugin)
    assert isinstance(instance, maibot_sdk.MaiBotPlugin)


def test_lifecycle_methods_implemented():
    import plugin

    instance = plugin.create_plugin()
    for name in ("on_load", "on_unload", "on_config_update"):
        method = getattr(instance, name, None)
        assert callable(method)


def test_config_model_has_required_sections():
    import plugin

    default = plugin.JmcomicPlugin.build_default_config()
    assert "plugin" in default
    assert default["plugin"]["config_version"]
    assert "features" in default


def test_no_action_decorator_used():
    source = (PLUGIN_DIR / "plugin.py").read_text(encoding="utf-8")
    assert "@Action" not in source
    assert "@Tool(" in source
    assert "@Command(" in source
