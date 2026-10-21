"""验证 /jm 命令的串行化与 PDF 清理不阻塞队列。

需要 maibot_sdk（MaiBot 自带的 python-env 里就有），因此不依赖 pytest，
直接运行即可：

    <MaiBot>/python-env/python.exe tests/test_queue_serial.py

关键断言：
1. 正在发送时提交的下一个命令返回「已加入队列」而不是被拒绝；
2. 第二个命令的发送必须在第一个命令**发送完成之后**才开始（串行）；
3. 第一个任务结束后，队列不因 PDF 保留期（_PDF_KEEP_SECONDS）而空等。
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

try:
    import maibot_sdk  # noqa: F401
except ImportError:  # pragma: no cover - 没装 SDK 时跳过
    print("跳过：未安装 maibot_sdk")
    sys.exit(0)

# 加载 plugin.py 时不要往插件目录写 __pycache__。
# 只靠命令行的 -B 不够：spec_from_file_location 的 loader 仍会缓存字节码。
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent.parent
PLUGIN_PATH = HERE / "plugin.py"
# 刻意不用插件目录下的路径：测试不该往安装目录里写东西
WORK_ROOT = Path(tempfile.mkdtemp(prefix="jm_queue_test_"))


def load_plugin():
    spec = importlib.util.spec_from_file_location("jm_queue_under_test", PLUGIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：dataclasses 处理 @dataclass 时会去
    # sys.modules[cls.__module__] 查模块；MaiBot 的 plugin_loader 也是
    # 这么做的（plugin_loader.py:573），这里保持一致。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(mod, events: list[tuple]):
    # WORK_ROOT 由 mkdtemp 新建，天然是空的，不需清理旧内容
    # （某些受限环境下连自己建的临时目录都删不掉，这里不依赖删除权限）
    root = WORK_ROOT
    root.mkdir(parents=True, exist_ok=True)

    class FakeSend:
        async def text(self, text, stream_id):  # noqa: ANN001
            events.append(("text", text, time.monotonic()))

        async def image(self, *a, **k):  # noqa: ANN002, ANN003
            return True

        async def hybrid(self, *a, **k):  # noqa: ANN002, ANN003
            return True

        async def custom(self, *a, **k):  # noqa: ANN002, ANN003
            return True

    plugin = mod.JmcomicPlugin()
    # ctx / config 在基类里是只读 property，底层字段分别是
    # _ctx 与 _plugin_config_instance（见 maibot_sdk/plugin.py:69）
    plugin._ctx = SimpleNamespace(
        logger=logging.getLogger("jm_queue_test"),
        send=FakeSend(),
        paths=SimpleNamespace(runtime_dir=root),
    )
    cfg = mod.JmcomicConfig()
    cfg.plugin.enabled = True
    cfg.components.enable_command = True
    cfg.features.cooldown_seconds = 0  # 冷却另行测试，这里不干扰
    cfg.features.max_pending_jobs = 2
    plugin._plugin_config_instance = cfg

    class FakeClient:
        def resolve_chapter(self, album_id, chapter_index=1):  # noqa: ANN001
            events.append(("resolve", album_id, time.monotonic()))
            return f"作品{album_id}", f"photo{album_id}"

        def download_photo(self, photo_id):  # noqa: ANN001
            d = root / f"dl_{photo_id}"
            d.mkdir(exist_ok=True)
            (d / "001.jpg").write_bytes(b"x")
            return d

    plugin._get_client = lambda: FakeClient()

    async def fake_send_chapter(**kw):  # noqa: ANN003
        album = kw["album_id"]
        pdf = root / f"JM{album}_第{kw['chapter_index']}章_1.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        events.append(("send_start", album, time.monotonic()))
        await asyncio.sleep(0.3)  # 模拟生成 + 发送耗时
        events.append(("send_done", album, time.monotonic()))
        return [pdf]

    plugin._send_chapter = fake_send_chapter
    return plugin, root


def stamp(events, kind, album):
    for name, tag, ts in events:
        if name == kind and tag == album:
            return ts
    raise AssertionError(f"事件缺失: {kind} {album}")


async def main() -> int:
    mod = load_plugin()
    # 故意设成 10 秒：若延迟清理仍 await 在任务里，第二个命令要等 10 秒
    mod._PDF_KEEP_SECONDS = 10.0

    events: list[tuple] = []
    plugin, _root = build(mod, events)

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        if not cond:
            ok = False

    print("场景：第一个命令正在发送时提交第二个命令")

    r1 = await plugin.handle_jm(
        stream_id="s1", matched_groups={"args": "111"}, user_id="u1"
    )
    await asyncio.sleep(0.05)  # 让 worker 取走第一个任务并进入发送
    r2 = await plugin.handle_jm(
        stream_id="s1", matched_groups={"args": "222"}, user_id="u2"
    )

    r3 = await plugin.handle_jm(
        stream_id="s1", matched_groups={"args": "333"}, user_id="u3"
    )
    r4 = await plugin.handle_jm(
        stream_id="s1", matched_groups={"args": "444"}, user_id="u4"
    )

    check("第一个命令被受理", r1 == (True, "已受理", 2), str(r1))
    check("发送中提交的命令进入队列", r2 == (True, "已加入队列", 2), str(r2))
    check("队列未满时继续排队", r3 == (True, "已加入队列", 2), str(r3))
    check("队列达到上限后拒绝", r4 == (True, "队列已满", 2), str(r4))

    queued_notice = [t for k, t, _ in events if k == "text" and "待处理队列" in t]
    check("发出了排队提示", bool(queued_notice), str(queued_notice))
    full_notice = [t for k, t, _ in events if k == "text" and "队列已满" in t]
    check("发出了队列已满提示", bool(full_notice), str(full_notice))

    await asyncio.wait_for(plugin._job_queue.join(), timeout=15)

    d1 = stamp(events, "send_done", "111")
    s2 = stamp(events, "send_start", "222")
    check("第二个命令在第一个发送完成后才开始（串行）", s2 >= d1,
          f"间隔 {s2 - d1:.3f}s")

    gap = s2 - d1
    check("队列未被 PDF 保留期阻塞", gap < 1.0,
          f"间隔 {gap:.3f}s（保留期设为 10s，若阻塞会是 10s）")

    # 再确认受理的任务都真的发出去了（111 正在发 + 222/333 排队 = 3 个）
    starts = [t for k, t, _ in events if k == "send_start"]
    check("受理的任务都执行了发送", len(starts) == 3, f"send_start 次数={len(starts)}")

    print("\n事件时间线：")
    t0 = events[0][2]
    for name, tag, ts in events:
        print(f"  +{ts - t0:6.2f}s  {name:12s} {tag}")

    print(f"\n{'全部通过' if ok else '存在失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    finally:
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
