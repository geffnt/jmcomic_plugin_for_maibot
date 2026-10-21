"""联网验证插件能否正常访问禁漫接口（UTF-8 BOM 兼容的回归测试）。

需要 jmcomic 与网络，故不在纯标准库用例里，按需手动运行：

    <MaiBot>/python-env/python.exe tests/test_jmcomic_live.py

只做「建客户端 + 解析作品」，不下载图片，正常情况下几秒结束。
网络不通时跳过（返回 0）；但若又出现「请求不是json格式」，说明 BOM 兼容失效，判失败。
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

try:
    import maibot_sdk  # noqa: F401
except ImportError:  # pragma: no cover
    print("跳过：未安装 maibot_sdk")
    sys.exit(0)

# 加载 plugin.py 时不要往插件目录写 __pycache__
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent.parent
PLUGIN_PATH = HERE / "plugin.py"
ALBUM_ID = "1179791"  # 短篇单话作品，用于验证接口连通


def load_plugin():
    spec = importlib.util.spec_from_file_location("jm_live_under_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_plugin()

    patched = module._install_bom_tolerance()
    print(f"1) BOM 兼容补丁: {patched or '已安装过'}")

    # 直接构造一个响应对象，验证剥离逻辑对真实结构有效
    class FakeResp:
        def __init__(self, content: bytes) -> None:
            self.content = content

        @property
        def text(self) -> str:
            if not hasattr(self, "_text"):
                self._text = self.content.decode("utf-8")
            return self._text

    probe = FakeResp(b'\xef\xbb\xbf{"code":200}')
    assert module._strip_leading_bom(probe) is True
    assert probe.text.startswith("{")
    print("2) 剥离逻辑自检: 通过")

    client = module.JmcomicClient(base_dir=tempfile.mkdtemp(prefix="jmlive_"))
    print(f"3) 解析作品 {ALBUM_ID} ...")
    try:
        title, photo_id = client.resolve_chapter(ALBUM_ID, 1)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if "请求不是json格式" in message:
            print("   >>> 失败：BOM 兼容失效，接口报文又被判成非 JSON")
            print("   ", message.splitlines()[0][:200])
            return 1
        print(f"   >>> 跳过：接口不可达（{type(exc).__name__}）")
        print("   ", message.splitlines()[0][:200])
        return 0

    print(f"   作品名: {title}")
    print(f"   photo_id: {photo_id}")
    if not title:
        print("   >>> 失败：作品名为空")
        return 1

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
