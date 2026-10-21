"""校验 PDF 分卷规则与提示文本（无需安装 maibot_sdk）。

``plugin.py`` 顶部要 ``import maibot_sdk``，没法直接导入，所以这里用 ``ast``
把纯函数抽出来单独编译执行，只测规则本身。
"""

import ast
from pathlib import Path

PLUGIN_PATH = Path(__file__).resolve().parent.parent / "plugin.py"


def _load_function(name: str):
    """从 plugin.py 源码中抽取指定模块级函数对象。

    同时执行模块级的全大写常量赋值（如 ``_UTF8_BOM``），因为被抽取的纯函数
    可能引用它们；这样测试不会和插件里的常量各自定义一份而漂移。
    """
    tree = ast.parse(PLUGIN_PATH.read_text(encoding="utf-8"))
    namespace: dict = {}
    consts = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
    ]
    if consts:
        const_module = ast.Module(body=consts, type_ignores=[])
        exec(compile(const_module, str(PLUGIN_PATH), "exec"), namespace)  # noqa: S102

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            # 补一个 future import：被抽出的片段没有插件顶部的
            # `from __future__ import annotations`，注解会被立即求值，
            # 遇到 Any / Path 这类名字就会 NameError
            future = ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
            ast.copy_location(future, node)
            module = ast.Module(body=[future, node], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(compile(module, str(PLUGIN_PATH), "exec"), namespace)  # noqa: S102
            return namespace[name]
    raise AssertionError(f"plugin.py 中找不到 {name}")


_resolve = _load_function("_resolve_pdf_batch_size")
_notice = _load_function("_format_page_notice")
_download_notice = _load_function("_format_download_notice")
_queue_notice = _load_function("_format_queue_notice")
_strip_bom = _load_function("_strip_leading_bom")
_failure_notice = _load_function("_format_failure_notice")


def _pdf_count(total: int, single_max: int = 60, per_pdf: int = 40) -> int:
    """按分卷规则算出会发几个 PDF。"""
    size = _resolve(total, single_max, per_pdf)
    return 0 if size <= 0 else -(-total // size)


def _notice_for(total: int, single_max: int = 60, per_pdf: int = 40) -> str:
    """按默认配置算出该给用户发什么提示。"""
    return _notice(total, _pdf_count(total, single_max, per_pdf))


# --------------------------------------------------------------------------- #
# 分卷规则
# --------------------------------------------------------------------------- #
def test_below_threshold_is_single_pdf():
    assert _resolve(30, 60, 40) == 30
    assert _pdf_count(30) == 1


def test_at_threshold_is_single_pdf():
    assert _resolve(60, 60, 40) == 60
    assert _pdf_count(60) == 1


def test_just_above_threshold_splits():
    assert _resolve(61, 60, 40) == 40
    assert _pdf_count(61) == 2  # 40 + 21


def test_large_chapter_splits_by_40():
    assert _pdf_count(100) == 3  # 40 + 40 + 20
    assert _pdf_count(160) == 4
    assert _pdf_count(200) == 5


def test_zero_images_per_pdf_means_no_split():
    assert _resolve(100, 60, 0) == 100
    assert _pdf_count(100, 60, 0) == 1


def test_empty_chapter():
    assert _resolve(0, 60, 40) == 0
    assert _pdf_count(0) == 0


def test_custom_thresholds():
    assert _pdf_count(50, 20, 10) == 5
    assert _pdf_count(20, 20, 10) == 1


# --------------------------------------------------------------------------- #
# 提示文本：单卷只说页数，分卷才说张数与份数
# --------------------------------------------------------------------------- #
def test_single_pdf_notice_reports_pages_only():
    assert _notice_for(1) == "共 1 页"
    assert _notice_for(30) == "共 30 页"
    assert _notice_for(60) == "共 60 页"


def test_split_notice_reports_images_and_pdf_count():
    assert _notice_for(61) == "共 61 张图片，将分为 2 个 PDF 发送"
    assert _notice_for(100) == "共 100 张图片，将分为 3 个 PDF 发送"
    assert _notice_for(200) == "共 200 张图片，将分为 5 个 PDF 发送"


def test_notice_switches_exactly_at_the_split_boundary():
    """59/60 说页数，61 起才说份数。"""
    assert _notice_for(59).endswith("页")
    assert _notice_for(60).endswith("页")
    assert "个 PDF 发送" in _notice_for(61)


def test_notice_ignores_pdf_count_when_not_splitting():
    """即使配置成不拆分，也只报页数。"""
    assert _notice_for(300, 60, 0) == "共 300 页"


# --------------------------------------------------------------------------- #
# 开始下载的提示：作品名用书名号
# --------------------------------------------------------------------------- #
def test_download_notice_uses_book_title_marks():
    assert _download_notice("某作品", 1) == "正在下载《某作品》第1章"
    assert _download_notice("Another", 12) == "正在下载《Another》第12章"


def test_download_notice_does_not_use_parentheses():
    text = _download_notice("某作品", 3)
    assert "（" not in text and "）" not in text
    assert "《" in text and "》" in text


# --------------------------------------------------------------------------- #
# 排队提示
# --------------------------------------------------------------------------- #
def test_queue_notice_reports_how_many_ahead():
    assert _queue_notice(1) == "已加入待处理队列（前面还有 1 个），等当前任务发完自动开始~"
    assert _queue_notice(3) == "已加入待处理队列（前面还有 3 个），等当前任务发完自动开始~"


def test_queue_notice_mentions_waiting_not_rejection():
    """排队提示要让人知道命令已受理，而不是被拒绝。"""
    text = _queue_notice(2)
    assert "待处理队列" in text
    assert "自动开始" in text


# --------------------------------------------------------------------------- #
# 禁漫接口的 UTF-8 BOM 兼容
# --------------------------------------------------------------------------- #
class _FakeResp:
    """模拟 curl_cffi 的响应：content 是普通属性，text 带缓存。"""

    def __init__(self, content: bytes) -> None:
        self.content = content

    @property
    def text(self) -> str:
        if not hasattr(self, "_text"):
            self._text = self.content.decode("utf-8")
        return self._text


def test_strip_bom_removes_leading_bom():
    resp = _FakeResp(b'\xef\xbb\xbf{"code":200}')
    assert _strip_bom(resp) is True
    assert resp.content == b'{"code":200}'


def test_strip_bom_invalidates_cached_text():
    """text 被缓存过也必须失效，否则校验读到的仍是带 BOM 的旧值。

    这是实际踩过的坑：只改 content 不改 _text，补丁等于没打。
    """
    resp = _FakeResp(b'\xef\xbb\xbf{"code":200}')
    assert resp.text[0] == "\ufeff"  # 先触发并缓存 text
    _strip_bom(resp)
    assert resp.text == '{"code":200}'
    assert ord(resp.text[0]) == 123  # '{'


def test_strip_bom_is_noop_without_bom():
    resp = _FakeResp(b'{"code":200}')
    assert _strip_bom(resp) is False
    assert resp.content == b'{"code":200}'


def test_strip_bom_is_idempotent():
    resp = _FakeResp(b'\xef\xbb\xbf{"code":200}')
    assert _strip_bom(resp) is True
    assert _strip_bom(resp) is False


def test_strip_bom_tolerates_unexpected_objects():
    """老版本 requests 的响应结构不同，也不能因此抛错。"""

    class Weird:
        content = "not bytes"

    assert _strip_bom(Weird()) is False
    assert _strip_bom(object()) is False


def test_strip_bom_falls_back_to_content_attribute():
    """content 只读时要回退去写 _content（requests 的存储字段）。"""

    class ReadOnly:
        def __init__(self, content: bytes) -> None:
            self._content = content
            self._text = content.decode("utf-8")

        @property
        def content(self) -> bytes:
            return self._content

    resp = ReadOnly(b'\xef\xbb\xbf{"a":1}')
    assert _strip_bom(resp) is True
    assert resp.content == b'{"a":1}'


# --------------------------------------------------------------------------- #
# 失败提示的长度控制
# --------------------------------------------------------------------------- #
def test_failure_notice_keeps_only_first_line():
    """jmcomic 会把 24 次重试详情塞进异常文本，不能整段发到聊天里。"""
    err = RuntimeError(
        "请求重试全部失败: [/setting]\n失败详情:\n  1. https://a/setting: xx\n  2. ..."
    )
    text = _failure_notice(err)
    assert "失败详情" not in text
    assert "请求重试全部失败" in text
    assert "禁漫接口暂时不可用" in text
    assert len(text.splitlines()) == 2


def test_failure_notice_truncates_long_head():
    text = _failure_notice(RuntimeError("x" * 500))
    assert len(text) <= 200
    assert text.endswith("…")


def test_failure_notice_passes_through_short_error():
    assert _failure_notice(ValueError("章节号超出范围（1-3）")) == (
        "下载或发送失败：章节号超出范围（1-3）"
    )


def test_failure_notice_handles_empty_message():
    assert _failure_notice(RuntimeError("")) == "下载或发送失败：RuntimeError"
