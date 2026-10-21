"""校验 _manifest.json 的合法性（无需安装 maibot_sdk）。"""

import json
import re
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "_manifest.json"

_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+)+$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _load() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_manifest_is_valid_json():
    data = _load()
    assert isinstance(data, dict)


def test_required_fields():
    data = _load()
    required = {
        "manifest_version",
        "id",
        "version",
        "name",
        "description",
        "author",
        "license",
        "urls",
        "host_application",
        "sdk",
        "capabilities",
        "i18n",
    }
    assert required.issubset(data.keys())


def test_manifest_version():
    assert _load()["manifest_version"] == 2


def test_id_format():
    assert _ID_RE.match(_load()["id"])


def test_version_is_semver():
    assert _SEMVER_RE.match(_load()["version"])


def test_version_ranges():
    data = _load()
    for section in ("host_application", "sdk"):
        assert _SEMVER_RE.match(data[section]["min_version"])
        assert _SEMVER_RE.match(data[section]["max_version"])


def test_urls_are_http():
    data = _load()
    assert data["author"]["url"].startswith(("http://", "https://"))
    assert data["urls"]["repository"].startswith(("http://", "https://"))


def test_dependencies_include_jmcomic():
    data = _load()
    packages = [d["name"] for d in data.get("dependencies", []) if d.get("type") == "python_package"]
    assert "jmcomic" in packages


def test_capabilities_cover_used_send_methods():
    """manifest 声明的能力必须覆盖 plugin.py 里用到的每个 ``ctx.send.*`` 方法。

    能力授权是精确字符串匹配（``AuthorizationManager.check_capability``），
    没有前缀或伞形展开，少声明一个就会在运行时被
    ``E_CAPABILITY_DENIED`` 拒绝。这里直接把源码用到的发送方法
    与清单声明做比对，避免再犯同类错误。
    """
    source = (MANIFEST_PATH.parent / "plugin.py").read_text(encoding="utf-8")
    used = set(re.findall(r"ctx\.send\.([a-z_]+)\(", source))
    declared = set(_load()["capabilities"])
    missing = {f"send.{name}" for name in used} - declared
    assert not missing, f"manifest 缺少能力声明: {sorted(missing)}"


def test_default_locale():
    assert _load()["i18n"]["default_locale"] == "zh-CN"
