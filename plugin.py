"""JMComic 漫画插件（MaiBot 新插件系统版）。

把 JMComic-Crawler-Python (https://github.com/hect0x7/JMComic-Crawler-Python)
接入 MaiBot：支持漫画搜索、详情查询、章节下载，以及 PDF / 图片两种方式发送。

- ``/jm`` 命令：下载并发送指定漫画的章节
- ``@Tool`` jmcomic_search_album：按关键字搜索漫画
- ``@Tool`` jmcomic_get_album_detail：查询漫画详情

本插件只使用 maibot-plugin-sdk，不修改 MaiBot 主程序。为保持改动边界清晰、
并符合官方「单文件 plugin.py」结构，全部实现都放在本文件中。
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maibot_sdk import (
    CONFIG_RELOAD_SCOPE_SELF,
    Command,
    Field,
    MaiBotPlugin,
    PluginConfigBase,
    Tool,
)
from maibot_sdk.types import ToolParameterInfo, ToolParamType

logger = logging.getLogger("maibot.plugin.jmcomic")

# 支持的图片后缀
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp")

# 每个合并转发（图文混合）消息内最多包含的图片数，避免单条消息过大
_HYBRID_BATCH_SIZE = 20

# 已发送的 PDF 保留多久再删除（秒）。
# 文件消息传给平台的是本地路径（NapCat 判定为 Local 后直接读取），
# 发完立刻删可能让上传读不到内容，所以留一段宽限期。
_PDF_KEEP_SECONDS = 300

# --------------------------------------------------------------------------- #
# PDF 体积控制
# --------------------------------------------------------------------------- #
# 目标：一个 60 页的 PDF 体积在 10 MB 左右，同时尽量保留画质。
# 每页 JPEG 预算 = PDF_TARGET_MB / PDF_PAGES_PER_TARGET ≈ 170 KB。
PDF_PAGES_PER_TARGET = 60
PDF_TARGET_MB = 10.0
PDF_MAX_RESOLUTION = 1500  # 长边最大像素
PDF_JPEG_QUALITY = 75  # 起始 JPEG 质量（不超预算时就用它，画质最好）
PDF_MIN_JPEG_QUALITY = 40  # 质量下限
PDF_MIN_RESOLUTION = 900  # 长边最小像素

_COMMAND_HELP = """JMComic 漫画插件使用说明：
/jm <漫画ID> [章节号] [pdf/jpg] - 下载并发送漫画
  例如：/jm 123456          （第 1 章，PDF）
        /jm 123456 2        （第 2 章，PDF）
        /jm 123456 2 jpg    （第 2 章，图片）
/jm help - 显示本帮助
也可以直接用「jm漫画」或「禁漫」开头触发。

⚠️ 群聊大量发图易被平台封号，强烈建议在私聊使用，或使用默认的 PDF 方式。"""


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def _to_int(value: Any, default: int = 0) -> int:
    """把 jmcomic 返回的字符串/数字安全转为 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_pdf_batch_size(
    total: int,
    single_pdf_max_pages: int = 60,
    images_per_pdf: int = 40,
) -> int:
    """决定每个 PDF 装多少页。

    分卷规则：

    - 总页数不超过 ``single_pdf_max_pages`` 时**合成单个 PDF**（返回 ``total``）；
    - 超过时按 ``images_per_pdf`` 拆成多个，``images_per_pdf`` 为 0 表示不拆分。

    Args:
        total: 图片总页数。
        single_pdf_max_pages: 单 PDF 的页数阈值，默认 60。
        images_per_pdf: 拆分时每份的页数，默认 40。

    Returns:
        int: 每份 PDF 包含的页数；``total`` 为 0 时返回 0。
    """
    if total <= 0:
        return 0
    if total <= single_pdf_max_pages:
        return total
    return images_per_pdf if images_per_pdf > 0 else total


def _format_page_notice(total: int, pdf_count: int) -> str:
    """生成发送 PDF 之前的提示文本。

    只有真的要分卷（``pdf_count > 1``）时才报张数与份数，
    单卷只报页数，避免啰嗦。

    Args:
        total: 图片总页数。
        pdf_count: 将要发送的 PDF 份数。

    Returns:
        str: 提示文本。
    """
    if pdf_count > 1:
        return f"共 {total} 张图片，将分为 {pdf_count} 个 PDF 发送"
    return f"共 {total} 页"


def _format_download_notice(title: str, chapter_index: int) -> str:
    """生成开始下载前的提示文本（作品名用书名号）。

    Args:
        title: 作品名。
        chapter_index: 章节序号。

    Returns:
        str: 提示文本。
    """
    return f"正在下载《{title}》第{chapter_index}章"


def _format_queue_notice(waiting: int) -> str:
    """生成「已加入待处理队列」的提示文本。

    Args:
        waiting: 排在该命令之前的任务数。

    Returns:
        str: 提示文本。
    """
    return f"已加入待处理队列（前面还有 {waiting} 个），等当前任务发完自动开始~"


def _format_failure_notice(error: BaseException, limit: int = 180) -> str:
    """把异常压成一条能直接发出去的短提示。

    jmcomic 的 ``RequestRetryAllFailException`` 会把 24 次重试的全部详情塞进
    ``str(异常)``，原样发到聊天里会刷屏（近 20 行），所以只保留首行并按需截断，
    完整堆栈仍写进日志。
    """
    raw = str(error).strip()
    head = raw.splitlines()[0].strip() if raw else type(error).__name__
    if len(head) > limit:
        head = head[:limit].rstrip() + "…"
    notice = f"下载或发送失败：{head}"
    if "请求重试全部失败" in raw or "请求不是json格式" in raw:
        notice += "\n（禁漫接口暂时不可用，请稍后再试）"
    return notice


@dataclass(slots=True)
class JmJob:
    """一次待处理的 ``/jm`` 请求。

    命令会先入队、再由唯一的 worker 串行取出执行，因此前一个任务的
    PDF 完全发完之后才会开始下一个。所有字段都是入队那一刻的快照，
    这样排队期间改配置也不会让已受理的命令中途换参数。
    """

    album_id: str
    chapter_index: int
    send_as_pdf: bool
    images_per_pdf: int
    max_pages: int
    use_forward: bool
    stream_id: str
    user_id: str


# --------------------------------------------------------------------------- #
# 禁漫接口的 UTF-8 BOM 兼容
# --------------------------------------------------------------------------- #
_UTF8_BOM = b"\xef\xbb\xbf"
_BOM_PATCH_MARK = "_maibot_jmcomic_bom_patched"


def _strip_leading_bom(resp: Any) -> bool:
    """去掉响应体开头的 UTF-8 BOM，返回是否真的改动过。

    2026-10 起禁漫服务端会在响应体前加一个 UTF-8 BOM（``b'\\xef\\xbb\\xbf'``），
    而 jmcomic 的报文校验只跳过空格 / 换行 / 制表符，于是一律判成
    「请求不是json格式」并重试到全部失败。jmcomic 2.7.7 已是最新版，
    上游尚未处理，所以只能在插件侧兼容。
    """
    try:
        content = getattr(resp, "content", None)
    except Exception:  # noqa: BLE001 - 流式响应读 content 可能抛错
        return False
    if not isinstance(content, (bytes, bytearray)) or not content.startswith(_UTF8_BOM):
        return False

    stripped = bytes(content[len(_UTF8_BOM):])
    try:
        # curl_cffi：content 是普通实例属性，可直接赋值
        resp.content = stripped
    except Exception:  # noqa: BLE001 - requests：content 是只读 property
        try:
            resp._content = stripped
        except Exception:  # noqa: BLE001
            return False

    # text 是带缓存的 property（curl_cffi 缓存到 _text）。不失效的话，
    # 校验读到的仍是带 BOM 的旧值，补丁等于没打——这一点踩过坑，务必保留。
    try:
        delattr(resp, "_text")
    except Exception:  # noqa: BLE001
        pass
    return True


def _install_bom_tolerance() -> list[str]:
    """给 jmcomic 打上 BOM 容错补丁，返回被修补的类名列表。

    补丁只做一件事：在 jmcomic 校验响应之前剥掉前导 BOM。它属于「放宽输入」
    而非改变行为，并且幂等，因此插件卸载后不还原——还原反而会影响同样在用
    jmcomic 的其它插件以及卸载瞬间的在途请求。
    """
    patched: list[str] = []
    try:
        from jmcomic.jm_client_impl import AbstractJmClient, JmApiClient
    except Exception as exc:  # noqa: BLE001 - 没装依赖时不应影响插件加载
        logger.debug("跳过禁漫 BOM 兼容补丁：%s", exc)
        return patched

    for cls in (AbstractJmClient, JmApiClient):
        original = getattr(cls, "raise_if_resp_should_retry", None)
        if original is None or getattr(original, _BOM_PATCH_MARK, False):
            continue

        def wrapper(self, resp, is_image, _original=original):  # noqa: ANN001
            _strip_leading_bom(resp)
            return _original(self, resp, is_image)

        setattr(wrapper, _BOM_PATCH_MARK, True)
        cls.raise_if_resp_should_retry = wrapper
        patched.append(cls.__name__)

    try:
        from jmcomic.jm_async_client import AsyncJmApiClient

        original = getattr(AsyncJmApiClient, "_raise_if_resp_should_retry", None)
        if original is not None and not getattr(original, _BOM_PATCH_MARK, False):

            def async_wrapper(resp, _original=original):  # noqa: ANN001
                _strip_leading_bom(resp)
                return _original(resp)

            setattr(async_wrapper, _BOM_PATCH_MARK, True)
            # 原方法是 staticmethod，赋值时也要包一层
            AsyncJmApiClient._raise_if_resp_should_retry = staticmethod(async_wrapper)
            patched.append(AsyncJmApiClient.__name__)
    except Exception as exc:  # noqa: BLE001
        logger.debug("异步客户端 BOM 补丁跳过：%s", exc)

    return patched


# --------------------------------------------------------------------------- #
# JMComic 封装层
# --------------------------------------------------------------------------- #
class JmcomicClient:
    """封装 jmcomic 的搜索、详情与章节下载。

    通过 ``JmOption`` 统一处理下载目录、代理、自定义域名与登录。
    客户端惰性初始化，首次真正使用前不会 import jmcomic。

    注意：本类所有方法都是阻塞的（会发起网络请求 / 下载文件），
    调用方必须使用 ``asyncio.to_thread`` 包装。
    """

    def __init__(
        self,
        base_dir: str,
        domain: str | None = None,
        proxy: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.domain = (domain or "").strip() or None
        self.proxy = (proxy or "").strip() or None
        self.username = (username or "").strip() or None
        self.password = (password or "").strip() or None

        self._option: Any = None
        self._client: Any = None
        self._lock = threading.Lock()

    # ------------------------------ 初始化 ------------------------------ #
    def _ensure_initialized(self) -> None:
        """惰性初始化 jmcomic 客户端（线程安全，可重复调用）。"""
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return

            try:
                import jmcomic  # 延迟导入，避免未安装依赖时影响插件加载
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 jmcomic 依赖，请在插件管理中确认依赖安装完成后再使用"
                ) from exc

            # 降低 jmcomic 的日志噪音，保留警告与错误
            logging.getLogger("jmcomic").setLevel(logging.WARNING)

            # 兜底再装一次 BOM 兼容（幂等；on_load 正常执行时这里会直接跳过）
            _install_bom_tolerance()

            option = jmcomic.JmOption.default()
            # 指定下载根目录（避免旧版 os.chdir 影响整个进程）
            option.dir_rule.base_dir = str(self.base_dir)

            client_kwargs: dict[str, Any] = {}
            if self.domain:
                client_kwargs["domain_list"] = [self.domain]
            if self.proxy:
                client_kwargs["proxies"] = self.proxy
            client = option.new_jm_client(**client_kwargs)

            if self.username and self.password:
                client.login(self.username, self.password)

            self._option = option
            self._client = client

    # ------------------------------ 搜索 ------------------------------ #
    def search_album(
        self, keyword: str, page: int = 1, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """按关键字搜索漫画，返回结果列表。"""
        self._ensure_initialized()
        client = self._client
        search_page = client.search_site(search_query=keyword, page=page)

        items: list[dict[str, Any]] = []
        for album_id, title, tags in search_page.iter_id_title_tag():
            items.append(
                {
                    "id": str(album_id),
                    "title": str(title),
                    "tags": [str(t) for t in (tags or [])],
                }
            )
            if len(items) >= max_results:
                break
        return items

    # ------------------------------ 详情 ------------------------------ #
    def get_album_detail(self, album_id: str) -> dict[str, Any]:
        """获取漫画详情（标题、作者、标签、章节列表等）。"""
        self._ensure_initialized()
        client = self._client
        album = client.get_album_detail(str(album_id))

        episodes: list[dict[str, Any]] = []
        for photo in album:
            episodes.append(
                {
                    "id": str(photo.photo_id),
                    "index": _to_int(getattr(photo, "album_index", 1), 1),
                    "title": str(photo.name),
                }
            )

        return {
            "id": str(album.album_id),
            "title": str(album.name),
            "authors": [str(a) for a in (album.authors or [])],
            "tags": [str(t) for t in (album.tags or [])],
            "works": [str(w) for w in (album.works or [])],
            "actors": [str(a) for a in (album.actors or [])],
            "page_count": _to_int(getattr(album, "page_count", 0)),
            "pub_date": str(getattr(album, "pub_date", "") or ""),
            "likes": str(getattr(album, "likes", "") or ""),
            "views": str(getattr(album, "views", "") or ""),
            "comment_count": _to_int(getattr(album, "comment_count", 0)),
            "episodes": episodes,
        }

    # ------------------------------ 下载 ------------------------------ #
    def resolve_chapter(self, album_id: str, chapter_index: int = 1) -> tuple[str, str]:
        """解析章节，返回 ``(作品名, photo_id)``。

        单独拆出来是为了让调用方能在**下载之前**先拿到作品名去提示用户，
        同时避免下载时再请求一次详情。
        """
        self._ensure_initialized()
        album = self._client.get_album_detail(str(album_id))
        episodes = list(album.episode_list or [])
        if not episodes:
            episodes = [(str(album_id), "1", str(album.name))]

        if chapter_index < 1 or chapter_index > len(episodes):
            raise ValueError(f"章节号超出范围（1-{len(episodes)}）")

        return str(album.name), str(episodes[chapter_index - 1][0])

    def download_photo(self, photo_id: str) -> Path | None:
        """下载单章图片，返回图片所在目录。"""
        self._ensure_initialized()
        result = self._option.download_photo(str(photo_id))

        # 新版本：DownloadResult.detail 是章节实体，带 save_path
        detail = getattr(result, "detail", None)
        save_path = getattr(detail, "save_path", None)
        if save_path and Path(save_path).is_dir():
            return Path(save_path)

        # 兼容旧版本：(photo, downloader) 元组返回值
        if isinstance(result, (tuple, list)) and result:
            legacy_path = getattr(result[0], "save_path", None)
            if legacy_path and Path(legacy_path).is_dir():
                return Path(legacy_path)

        # 兜底：在 base_dir 下查找包含图片的目录
        return self._find_downloaded_dir()

    def download_chapter(self, album_id: str, chapter_index: int = 1) -> Path | None:
        """下载指定章节的全部图片，返回图片所在目录。

        便捷封装：先解析章节再下载。需要作品名时请改用
        ``resolve_chapter`` + ``download_photo``，以免重复请求详情。
        """
        _, photo_id = self.resolve_chapter(album_id, chapter_index)
        return self.download_photo(photo_id)

    def _find_downloaded_dir(self) -> Path | None:
        """在 base_dir 下查找图片数量最多的目录。"""
        candidates: list[tuple[Path, int]] = []
        for item in self.base_dir.iterdir():
            if not item.is_dir():
                continue
            images = [
                f
                for f in item.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
            ]
            if images:
                candidates.append((item, len(images)))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: -pair[1])
        return candidates[0][0]


def create_pdf_from_images(
    image_files: list[Path],
    output_path: Path,
    album_title: str = "",
    max_resolution: int = PDF_MAX_RESOLUTION,
    jpeg_quality: int = PDF_JPEG_QUALITY,
    target_bytes_per_page: int = 0,
) -> bool:
    """把若干图片合并为一个 PDF，并按体积预算压缩。

    体积控制策略（优先保画质，其次保分辨率，最后才降质量）：

    1. 长边缩到 ``max_resolution``，按 ``jpeg_quality`` 编码；
       **不超预算就直接用**，此时画质最好。
    2. 超预算时按面积比例估算所需分辨率（JPEG 体积大致与像素数成正比），
       降分辨率后重编码，最多迭代 4 次。
    3. 分辨率已到 ``PDF_MIN_RESOLUTION`` 仍超预算，才逐步降质量，
       直到 ``PDF_MIN_JPEG_QUALITY``。

    ``target_bytes_per_page`` 为 0 表示不做体积控制（直接用 ``jpeg_quality``）。

    Args:
        image_files: 图片文件列表，按顺序成页。
        output_path: 输出 PDF 路径。
        album_title: PDF 标题。
        max_resolution: 长边最大像素。
        jpeg_quality: 起始 JPEG 质量。
        target_bytes_per_page: 每页 JPEG 的字节预算。

    Returns:
        bool: 是否成功生成；单张图片失败会跳过，不影响其余图片。
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        from PIL import Image
    except ImportError as exc:
        logger.error("缺少 reportlab / Pillow 依赖，无法生成 PDF：%s", exc)
        return False

    # 关掉 reportlab 的 ASCII85 编码。
    # 默认它会给内嵌 JPEG 外面再套一层 ASCII85（过滤器变成
    # ['ASCII85Decode', 'DCTDecode']），凭空多出约 25% 体积；
    # 关掉后过滤器只剩 DCTDecode，JPEG 原样嵌入，体积≈JPEG 本身，
    # 画质完全不受影响。实测 735,913 字节的 JPEG：1.252x -> 1.002x。
    # 该开关是 reportlab 的进程级配置，但只改变编码方式、不影响 PDF 有效性。
    try:
        from reportlab import rl_config

        rl_config.useA85 = 0
    except Exception:  # noqa: BLE001 - 设置失败不影响生成
        pass

    page_stats: list[tuple[int, int, int]] = []  # (长边像素, 质量, 字节)

    def _fit_longest_side(image: Any, longest: int) -> Any:
        """把图片长边缩到不超过 longest，短边等比缩放。"""
        width, height = image.size
        current = max(width, height)
        if longest <= 0 or current <= longest:
            return image
        scale = longest / current
        return image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )

    def _encode_jpeg(image: Any, quality: int) -> bytes:
        """把图片编码为 JPEG 字节。

        这里刻意用**基线 JPEG**（不加 ``progressive=True``）：
        reportlab 只会直接嵌入基线 JPEG（DCTDecode），渐进式的认不出来，
        会退化成解码后重编码，体积反而变大、画质也不再可控。
        """
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    def _compress_page(image: Any) -> bytes:
        """按预算压缩单页图片，返回 JPEG 字节。"""
        page = _fit_longest_side(image, max_resolution)
        quality = jpeg_quality
        data = _encode_jpeg(page, quality)

        if target_bytes_per_page > 0 and len(data) > target_bytes_per_page:
            # 阶段一：优先保质量，按面积比例降分辨率（JPEG 体积≈与像素数成正比）
            for _ in range(4):
                current = max(page.size)
                if current <= PDF_MIN_RESOLUTION:
                    break
                ratio = (target_bytes_per_page / len(data)) ** 0.5
                target = max(
                    PDF_MIN_RESOLUTION, int(current * max(0.5, min(0.95, ratio)))
                )
                if target >= current:
                    break
                page = _fit_longest_side(image, target)
                data = _encode_jpeg(page, quality)
                if len(data) <= target_bytes_per_page:
                    break

            # 阶段二：分辨率已到下限仍超预算，才降质量
            while len(data) > target_bytes_per_page and quality > PDF_MIN_JPEG_QUALITY:
                quality = max(PDF_MIN_JPEG_QUALITY, quality - 6)
                data = _encode_jpeg(page, quality)

        page_stats.append((max(page.size), quality, len(data)))
        return data

    try:
        pdf = canvas.Canvas(str(output_path), pagesize=A4)
        if album_title:
            pdf.setTitle(album_title)

        page_width, page_height = A4
        margin = 1 * cm
        avail_w = page_width - 2 * margin
        avail_h = page_height - 2 * margin

        first_page = True
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                if img.mode in ("RGBA", "LA", "P"):
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                data = _compress_page(img)
                # 压缩只做等比缩放，宽高比与原图一致，直接按原图比例排版
                draw_w, draw_h = img.size

                scale = min(avail_w / draw_w, avail_h / draw_h)
                draw_w *= scale
                draw_h *= scale
                x = (page_width - draw_w) / 2
                y = (page_height - draw_h) / 2

                if not first_page:
                    pdf.showPage()
                first_page = False
                # 直接嵌入已压好的 JPEG 字节，reportlab 不会再重新编码
                pdf.drawImage(
                    ImageReader(io.BytesIO(data)), x, y, width=draw_w, height=draw_h
                )
            except Exception as exc:  # noqa: BLE001 - 单图失败不应中断整个 PDF
                logger.warning("处理图片 %s 失败：%s", img_path, exc)
                continue

        if page_stats:
            sizes = [item[2] for item in page_stats]
            logger.info(
                "PDF 压缩统计：%d 页，单页 %.0f~%.0f KB（平均 %.0f KB），长边 %d~%d px，质量 %d~%d",
                len(page_stats),
                min(sizes) / 1024,
                max(sizes) / 1024,
                sum(sizes) / len(sizes) / 1024,
                min(item[0] for item in page_stats),
                max(item[0] for item in page_stats),
                min(item[1] for item in page_stats),
                max(item[1] for item in page_stats),
            )

        pdf.save()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("创建 PDF 失败：%s", exc, exc_info=True)
        return False


# --------------------------------------------------------------------------- #
# 配置模型
# --------------------------------------------------------------------------- #
class PluginSection(PluginConfigBase):
    """基础设置。"""

    __ui_label__ = "基础设置"
    __ui_icon__ = "settings"
    __ui_order__ = 0

    enabled: bool = Field(default=False, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本号")


class ComponentSection(PluginConfigBase):
    """组件开关。"""

    __ui_label__ = "组件"
    __ui_icon__ = "extension"
    __ui_order__ = 1

    enable_command: bool = Field(default=True, description="是否启用 /jm 下载命令")


class ApiSection(PluginConfigBase):
    """JMComic API。"""

    __ui_label__ = "JMComic API"
    __ui_icon__ = "language"
    __ui_order__ = 2

    domain: str = Field(default="", description="JMComic 自定义 API 域名（留空使用默认）")
    proxy: str = Field(
        default="",
        description="代理地址，如 http://127.0.0.1:7890（留空使用系统代理）",
    )


class DownloadSection(PluginConfigBase):
    """下载设置。"""

    __ui_label__ = "下载"
    __ui_icon__ = "download"
    __ui_order__ = 3

    base_dir: str = Field(default="", description="漫画下载目录（留空使用插件运行时临时目录）")


class AuthSection(PluginConfigBase):
    """登录认证。"""

    __ui_label__ = "登录认证"
    __ui_icon__ = "lock"
    __ui_order__ = 4

    username: str = Field(default="", description="JMComic 用户名（可选，用于访问受限内容）")
    password: str = Field(default="", description="JMComic 密码")


class FeaturesSection(PluginConfigBase):
    """功能设置。"""

    __ui_label__ = "功能"
    __ui_icon__ = "tune"
    __ui_order__ = 5

    cooldown_seconds: int = Field(default=30, description="命令冷却时间（秒），0 表示不限制")
    max_pending_jobs: int = Field(
        default=3,
        description="待处理队列上限：正在发送时最多还能排几个命令",
    )
    max_pages_per_send: int = Field(default=1000, description="单次最多发送页数")
    use_forward_message: bool = Field(default=True, description="以合并转发（图文混合）方式发送")
    send_as_pdf: bool = Field(default=True, description="将图片合并为 PDF 发送")
    pdf_fallback_to_images: bool = Field(
        default=False,
        description="PDF 发送失败时是否回退为逐张发送图片（默认关闭，失败会直接提示）",
    )
    single_pdf_max_pages: int = Field(
        default=60,
        description="页数不超过该值时合并为单个 PDF，超过则按 images_per_pdf 拆成多个",
    )
    images_per_pdf: int = Field(
        default=40, description="拆分时每个 PDF 包含的图片数量（0 表示不拆分，最大 60）"
    )
    pdf_target_mb_per_60_pages: float = Field(
        default=PDF_TARGET_MB,
        description="60 页 PDF 的目标体积（MB），按页均摊作为单页压缩预算；0 表示不限制",
    )
    pdf_max_resolution: int = Field(
        default=PDF_MAX_RESOLUTION, description="PDF 内图片长边最大像素"
    )
    pdf_jpeg_quality: int = Field(
        default=PDF_JPEG_QUALITY, description="PDF 内图片的起始 JPEG 质量（不超预算时使用）"
    )


class JmcomicConfig(PluginConfigBase):
    """JMComic 漫画插件配置。"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    components: ComponentSection = Field(default_factory=ComponentSection)
    api: ApiSection = Field(default_factory=ApiSection)
    download: DownloadSection = Field(default_factory=DownloadSection)
    auth: AuthSection = Field(default_factory=AuthSection)
    features: FeaturesSection = Field(default_factory=FeaturesSection)


# --------------------------------------------------------------------------- #
# 插件类
# --------------------------------------------------------------------------- #
class JmcomicPlugin(MaiBotPlugin):
    """JMComic 漫画插件。"""

    config_model = JmcomicConfig

    def __init__(self) -> None:
        super().__init__()
        self._cooldown_cache: dict[str, float] = {}
        self._client: JmcomicClient | None = None
        # 组件 RPC 默认 60 秒超时，而下载整章再合成 PDF 通常更久，
        # 所以命令只负责受理，实际工作放进后台任务。
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._pending_pdfs: set[Path] = set()
        # 命令串行化：正在发送时的后续命令进队列，等前一个 PDF 完全发完再执行。
        # 构造 asyncio.Queue 不会绑定事件循环（3.10+），因此放在 __init__ 是安全的。
        self._job_queue: asyncio.Queue[JmJob] = asyncio.Queue()
        self._worker_task: asyncio.Task[Any] | None = None
        self._busy = False

    # ---------------------------- 生命周期 ---------------------------- #
    async def on_load(self) -> None:
        """插件加载时执行。"""
        try:
            if importlib.util.find_spec("jmcomic") is None:
                self.ctx.logger.warning(
                    "未检测到 jmcomic 依赖，请确认插件依赖已安装（reportlab / Pillow 同理）"
                )
            else:
                patched = _install_bom_tolerance()
                self.ctx.logger.info(
                    "已验证禁漫接口的 UTF-8 BOM 兼容：%s",
                    "、".join(patched) if patched else "已是最新，无需处理",
                )
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning("安装禁漫 BOM 兼容补丁失败：%s", exc)
        self.ctx.logger.info("JMComic 漫画插件已加载")

    async def on_unload(self) -> None:
        """插件卸载时执行。"""
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()
        self._worker_task = None
        self._busy = False
        dropped = 0
        while True:
            try:
                self._job_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._job_queue.task_done()
            dropped += 1
        if dropped:
            self.ctx.logger.info("卸载时丢弃了 %d 个排队中的命令", dropped)
        for pdf_path in list(self._pending_pdfs):
            await self._cleanup_target(pdf_path)
        self._pending_pdfs.clear()
        self._cooldown_cache.clear()
        self._client = None
        self.ctx.logger.info("JMComic 漫画插件已卸载")

    async def on_config_update(
        self, scope: str, config_data: dict[str, Any], version: str
    ) -> None:
        """配置热重载时执行。"""
        del config_data
        if scope == CONFIG_RELOAD_SCOPE_SELF:
            # 配置变化后重建客户端，使域名 / 代理 / 下载目录等生效
            self._cooldown_cache.clear()
            self._client = None
            self.ctx.logger.info("JMComic 漫画插件配置已更新（version=%s）", version)

    # ---------------------------- 内部工具 ---------------------------- #
    def _get_client(self) -> JmcomicClient:
        """惰性创建 jmcomic 客户端。"""
        if self._client is None:
            base_dir = (self.config.download.base_dir or "").strip()
            if not base_dir:
                base_dir = str(self.ctx.paths.runtime_dir / "jmcomic_downloads")
            self._client = JmcomicClient(
                base_dir=base_dir,
                domain=self.config.api.domain or None,
                proxy=self.config.api.proxy or None,
                username=self.config.auth.username or None,
                password=self.config.auth.password or None,
            )
        return self._client

    @staticmethod
    def _extract_user_id(kwargs: dict[str, Any]) -> str:
        """尽力从命令上下文中提取用户标识，用于冷却控制。"""
        message = kwargs.get("message")
        if isinstance(message, dict):
            for key in ("user_id", "sender_id"):
                value = message.get(key)
                if value:
                    return str(value)
            info = message.get("message_info") or message.get("user_info") or {}
            if isinstance(info, dict):
                user_info = info.get("user_info") or {}
                if isinstance(user_info, dict) and user_info.get("user_id"):
                    return str(user_info["user_id"])
                if info.get("user_id"):
                    return str(info["user_id"])
        for key in ("user_id", "sender_id"):
            value = kwargs.get(key)
            if value:
                return str(value)
        return str(kwargs.get("stream_id") or "unknown")

    async def _cleanup_target(self, target: Path) -> None:
        """删除下载 / PDF 产生的临时文件，失败不抛出。"""
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning("清理临时文件 %s 失败：%s", target, exc)

    # --------------------------- 队列与调度 --------------------------- #
    def _pending_count(self) -> int:
        """当前未完成的命令数（正在发送的那个算 1 个）。"""
        return self._job_queue.qsize() + (1 if self._busy else 0)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        """把协程挂到后台，并纳入统一的取消管理。"""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _ensure_worker(self) -> None:
        """确保串行 worker 正在运行。"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = self._spawn(self._job_worker())

    async def _job_worker(self) -> None:
        """串行消费命令队列。

        每次都只取一个任务并等它彻底跑完（PDF 全部发送结束）才取下一个，
        这就是「前一个发完之前，后续命令只会排队」的实现方式。
        """
        while True:
            job = await self._job_queue.get()
            self._busy = True
            try:
                await self._run_download_job(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - 单个任务失败不应终止 worker
                self.ctx.logger.error("处理排队命令失败：%s", exc, exc_info=True)
            finally:
                self._busy = False
                self._job_queue.task_done()

    # ------------------------------ 命令 ------------------------------ #
    @Command(
        "jm",
        description="从 JMComic 下载并发送漫画",
        pattern=r"^(?:/jm|jm漫画|禁漫)\s*(?P<args>.*)$",
    )
    async def handle_jm(self, **kwargs: Any) -> tuple[bool, str, int]:
        """处理 /jm 下载命令。"""
        if not self.config.plugin.enabled or not self.config.components.enable_command:
            return False, "", 0

        stream_id = str(kwargs.get("stream_id", "") or "")
        matched = kwargs.get("matched_groups") or {}
        args_str = str(matched.get("args", "") or "").strip()

        if args_str.lower() in ("help", "帮助", "?", "？", ""):
            await self.ctx.send.text(_COMMAND_HELP, stream_id)
            return True, "显示帮助", 2

        args = args_str.split()
        album_id = args[0]
        chapter_index = 1
        output_format: str | None = None
        for arg in args[1:]:
            lower = arg.lower()
            if lower in ("pdf", "jpg", "jpeg", "png"):
                output_format = lower
            elif arg.isdigit():
                chapter_index = int(arg)

        if not album_id.isdigit():
            await self.ctx.send.text("请输入有效的漫画 ID，如：/jm 123456", stream_id)
            return True, "无效 ID", 2

        user_id = self._extract_user_id(kwargs)

        # 输出格式：命令参数优先，否则用配置默认值
        if output_format == "pdf":
            send_as_pdf = True
        elif output_format in ("jpg", "jpeg", "png"):
            send_as_pdf = False
        else:
            send_as_pdf = bool(self.config.features.send_as_pdf)

        # 参数在入队那一刻定下来，排队期间改配置不影响已受理的命令
        job = JmJob(
            album_id=album_id,
            chapter_index=chapter_index,
            send_as_pdf=send_as_pdf,
            images_per_pdf=max(0, min(60, int(self.config.features.images_per_pdf))),
            max_pages=max(1, int(self.config.features.max_pages_per_send)),
            use_forward=bool(self.config.features.use_forward_message),
            stream_id=stream_id,
            user_id=user_id,
        )

        # 正在发送（或已有排队）时，后续命令一律排队，
        # 等前一个 PDF 完全发完再由 worker 依次执行。
        pending = self._pending_count()
        if pending > 0:
            # 上限只约束「排队中」的数量，不把正在发送的那个算进去
            limit = max(1, int(self.config.features.max_pending_jobs))
            if self._job_queue.qsize() >= limit:
                await self.ctx.send.text(
                    f"待处理队列已满（{limit} 个），请等当前任务发完后再试~", stream_id
                )
                return True, "队列已满", 2
            self._cooldown_cache[user_id] = time.time()
            await self._job_queue.put(job)
            self._ensure_worker()
            await self.ctx.send.text(_format_queue_notice(pending), stream_id)
            return True, "已加入队列", 2

        # 空闲时才做冷却检查：否则冷却会把本该排队的命令直接挡掉
        cooldown = max(0, int(self.config.features.cooldown_seconds))
        now = time.time()
        last_use = self._cooldown_cache.get(user_id, 0.0)
        if cooldown > 0 and (now - last_use) < cooldown:
            remaining = int(cooldown - (now - last_use)) + 1
            await self.ctx.send.text(f"冷却中，请 {remaining} 秒后再试~", stream_id)
            return True, "冷却中", 2

        # 受理即记冷却，避免空闲时被连续触发
        self._cooldown_cache[user_id] = time.time()
        await self._job_queue.put(job)
        self._ensure_worker()
        # 提示信息由 worker 在拿到作品名后发送（见 _run_download_job）
        return True, "已受理", 2

    async def _run_download_job(self, job: JmJob) -> None:
        """执行单个排队命令的下载与发送。

        组件 RPC 默认超时 60 秒（``DEFAULT_COMPONENT_RPC_TIMEOUT_MS``），
        下载整章再合成 PDF 通常远超这个时间，所以命令只负责入队。

        语义上，本函数**返回即代表这一次的 PDF 已全部发送完毕**，
        worker 随后才会取下一个命令。因此 PDF 的延迟清理必须异步化
        （见 ``_schedule_pdf_cleanup``），否则 300 秒保留期会把队列堵死。
        """
        created_pdfs: list[Path] = []
        try:
            client = self._get_client()
            # 先解析出作品名，好在下载开始前就提示用户
            title, photo_id = await asyncio.to_thread(
                client.resolve_chapter, job.album_id, job.chapter_index
            )
            await self.ctx.send.text(
                _format_download_notice(title, job.chapter_index), job.stream_id
            )

            chapter_dir = await asyncio.to_thread(client.download_photo, photo_id)
            if not chapter_dir:
                await self.ctx.send.text(
                    "下载失败，请检查漫画 ID 和章节号是否正确", job.stream_id
                )
                return
            try:
                created_pdfs = await self._send_chapter(
                    chapter_dir=chapter_dir,
                    album_id=job.album_id,
                    chapter_index=job.chapter_index,
                    send_as_pdf=job.send_as_pdf,
                    images_per_pdf=job.images_per_pdf,
                    max_pages=job.max_pages,
                    use_forward=job.use_forward,
                    stream_id=job.stream_id,
                )
            finally:
                await self._cleanup_target(chapter_dir)
        except asyncio.CancelledError:
            self.ctx.logger.info("JMComic 后台任务已取消")
            raise
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("下载或发送失败：%s", exc, exc_info=True)
            await self.ctx.send.text(_format_failure_notice(exc), job.stream_id)
        finally:
            self._schedule_pdf_cleanup(created_pdfs)

    def _schedule_pdf_cleanup(self, pdf_paths: list[Path]) -> None:
        """把已发送 PDF 的延迟清理挂到独立后台任务。

        刻意不 await：``_PDF_KEEP_SECONDS`` 是 300 秒，若在这里等，
        队列里的下一个命令会白等 5 分钟才轮到。路径先同步登记进
        ``_pending_pdfs``，卸载时才能兜底清掉。
        """
        if not pdf_paths:
            return
        self._pending_pdfs.update(pdf_paths)
        self._spawn(self._cleanup_pdfs_later(list(pdf_paths)))

    async def _cleanup_pdfs_later(self, pdf_paths: list[Path]) -> None:
        """等宽限期结束后删除 PDF。

        文件消息传给平台的是本地路径，发完立刻删可能让上传读不到内容。
        """
        try:
            await asyncio.sleep(_PDF_KEEP_SECONDS)
        except asyncio.CancelledError:
            raise
        for pdf_path in pdf_paths:
            self._pending_pdfs.discard(pdf_path)
            await self._cleanup_target(pdf_path)

    # ---------------------------- 发送逻辑 ---------------------------- #
    async def _send_chapter(
        self,
        chapter_dir: Path,
        album_id: str,
        chapter_index: int,
        send_as_pdf: bool,
        images_per_pdf: int,
        max_pages: int,
        use_forward: bool,
        stream_id: str,
    ) -> list[Path]:
        """发送章节内容：PDF 或图片（合并转发 / 逐张）。

        Returns:
            list[Path]: 本次生成的 PDF 路径，供调用方延后清理；图片模式返回空列表。
        """
        image_files = sorted(
            f
            for f in chapter_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES
        )[:max_pages]

        if not image_files:
            await self.ctx.send.text("章节中没有找到图片", stream_id)
            return []

        if send_as_pdf:
            return await self._send_as_pdf(
                image_files, album_id, chapter_index, images_per_pdf, stream_id
            )

        if use_forward:
            await self._send_images_hybrid(image_files, stream_id)
        else:
            await self._send_images_separately(image_files, stream_id)
        return []

    async def _send_images_separately(self, image_files: list[Path], stream_id: str) -> None:
        """逐张发送图片，每张前附带页码说明。"""
        total = len(image_files)
        for index, image_file in enumerate(image_files, start=1):
            await self.ctx.send.text(f"第 {index}/{total} 页", stream_id)
            if not await self._send_image_file(image_file, stream_id):
                await self.ctx.send.text(f"第 {index} 页发送失败，已跳过", stream_id)

    async def _send_images_hybrid(self, image_files: list[Path], stream_id: str) -> None:
        """以合并转发（图文混合）方式分批发送图片。"""
        total = len(image_files)
        for start in range(0, total, _HYBRID_BATCH_SIZE):
            batch = image_files[start : start + _HYBRID_BATCH_SIZE]
            segments: list[dict[str, Any]] = []
            for offset, image_file in enumerate(batch, start=1):
                b64 = self._image_to_base64(image_file)
                if not b64:
                    continue
                segments.append({"type": "text", "content": f"第 {start + offset}/{total} 页"})
                segments.append({"type": "image", "content": b64})
            if segments:
                try:
                    await self.ctx.send.hybrid(segments, stream_id)
                except Exception as exc:  # noqa: BLE001
                    self.ctx.logger.error("发送图文混合消息失败：%s", exc, exc_info=True)

    async def _send_image_file(self, image_file: Path, stream_id: str) -> bool:
        """把单个图片文件转为 base64 并发送。"""
        b64 = self._image_to_base64(image_file)
        if not b64:
            return False
        try:
            return bool(await self.ctx.send.image(b64, stream_id))
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("发送图片失败：%s", exc, exc_info=True)
            return False

    async def _send_as_pdf(
        self,
        image_files: list[Path],
        album_id: str,
        chapter_index: int,
        images_per_pdf: int,
        stream_id: str,
    ) -> list[Path]:
        """把图片合并为 PDF 并发送。

        分卷规则：页数不超过 ``single_pdf_max_pages``（默认 60）时合成单个 PDF；
        超过时按 ``images_per_pdf``（默认 40）拆成多个 PDF 分别发送。

        Returns:
            list[Path]: 本次生成的 PDF 路径，供调用方延后清理。
        """
        total = len(image_files)
        batch_size = max(
            1,
            _resolve_pdf_batch_size(
                total,
                max(0, int(self.config.features.single_pdf_max_pages)),
                images_per_pdf,
            ),
        )

        pdf_count = -(-total // batch_size)  # 向上取整
        self.ctx.logger.info(
            "本次共 %d 页，%s",
            total,
            "合并为 1 个 PDF"
            if pdf_count == 1
            else f"拆分为 {pdf_count} 个 PDF（每个 {batch_size} 页）",
        )
        # 只有真的要分卷时才提示拆分数量，单卷只报页数
        await self.ctx.send.text(_format_page_notice(total, pdf_count), stream_id)

        created: list[Path] = []
        pdf_sent = True

        features = self.config.features
        max_resolution = max(200, int(features.pdf_max_resolution))
        jpeg_quality = max(1, min(95, int(features.pdf_jpeg_quality)))
        target_mb = float(features.pdf_target_mb_per_60_pages)
        # 按页均摊目标体积，得到单页 JPEG 预算（0 表示不限制）
        page_budget = (
            int(target_mb * 1024 * 1024 / PDF_PAGES_PER_TARGET) if target_mb > 0 else 0
        )
        self.ctx.logger.info(
            "PDF 压缩参数：长边 %dpx，起始质量 %d，单页预算 %.0f KB",
            max_resolution,
            jpeg_quality,
            page_budget / 1024,
        )

        for start in range(0, len(image_files), batch_size):
            batch = image_files[start : start + batch_size]
            pdf_name = f"JM{album_id}_第{chapter_index}章_{start // batch_size + 1}.pdf"
            pdf_path = self.ctx.paths.runtime_dir / pdf_name
            ok = await asyncio.to_thread(
                create_pdf_from_images,
                batch,
                pdf_path,
                f"JM{album_id} 第{chapter_index}章",
                max_resolution,
                jpeg_quality,
                page_budget,
            )
            if not ok or not pdf_path.exists():
                pdf_sent = False
                break
            created.append(pdf_path)
            size_mb = pdf_path.stat().st_size / 1024 / 1024
            self.ctx.logger.info(
                "%s 生成完成：%.2f MB（%d 页）", pdf_name, size_mb, len(batch)
            )
            if not await self._send_pdf_file(pdf_path, pdf_name, stream_id):
                pdf_sent = False
                break

        if not pdf_sent:
            if self.config.features.pdf_fallback_to_images:
                await self.ctx.send.text("PDF 发送失败，改为逐张发送图片...", stream_id)
                await self._send_images_separately(image_files, stream_id)
            else:
                await self.ctx.send.text(
                    "PDF 发送失败，本次未发送图片。"
                    "如需失败时回退为图片，请在配置中开启 features.pdf_fallback_to_images。",
                    stream_id,
                )
        return created

    async def _send_pdf_file(self, pdf_path: Path, pdf_name: str, stream_id: str) -> bool:
        """把 PDF 作为文件消息发送。

        这里传的是**本地绝对路径**而不是 base64，原因有两个：

        1. base64 得整个塞进一次 RPC 调用，而 Runner 的帧上限是
           ``MAX_FRAME_SIZE``（16 MB）；40 页 PDF 的 base64 约 29 MB，
           会直接抛 ``E_UNKNOWN 帧大小 ... 超过最大限制 16777216``。
           本地路径只是一个短字符串，与文件大小无关。
        2. NapCat 的 ``checkUriType`` 会先用 ``fs.existsSync`` 判定为
           Local 并直接使用该路径，既不下载、也不受下载代理影响。

        链路：``url`` 是 Host 与适配器**唯一都认**的键名；适配器的
        ``_normalize_file_reference`` 会补上 ``file://`` 前缀，
        NapCat 再 ``slice(7)`` 还原成本地路径。

        因此：插件与 NapCat 必须同机，且文件在发送期间必须存在
        （见 ``_PDF_KEEP_SECONDS``）。
        """
        try:
            return bool(
                await self.ctx.send.custom(
                    "file",
                    {"name": pdf_name, "url": pdf_path.resolve().as_posix()},
                    stream_id,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("发送 PDF 文件失败：%s", exc, exc_info=True)
            return False

    @staticmethod
    def _image_to_base64(image_file: Path) -> str:
        """读取图片并转为 base64，失败返回空串。"""
        try:
            return base64.b64encode(image_file.read_bytes()).decode()
        except Exception as exc:  # noqa: BLE001
            logger.error("读取图片 %s 失败：%s", image_file, exc)
            return ""

    # ------------------------------ 工具 ------------------------------ #
    @Tool(
        "jmcomic_search_album",
        brief_description="搜索禁漫天堂的漫画本子",
        detailed_description=(
            "按关键字搜索禁漫天堂的漫画，返回漫画 ID、标题与标签列表。"
            "使用场景：用户想找某部漫画时调用。参数 keyword 为必填搜索关键字，"
            "page 为可选页码（从 1 开始）。"
        ),
        parameters=[
            ToolParameterInfo(
                name="keyword",
                param_type=ToolParamType.STRING,
                description="搜索关键字",
                required=True,
            ),
            ToolParameterInfo(
                name="page",
                param_type=ToolParamType.INTEGER,
                description="页码，从 1 开始",
                required=False,
                default=1,
            ),
        ],
    )
    async def search_album_tool(
        self, keyword: str, page: int = 1, **kwargs: Any
    ) -> dict[str, Any]:
        """搜索漫画（供 LLM 调用）。"""
        del kwargs
        if not self.config.plugin.enabled:
            return {"success": False, "content": "插件未启用。"}

        keyword = (keyword or "").strip()
        if not keyword:
            return {"success": False, "content": "搜索关键字不能为空。"}

        try:
            client = self._get_client()
            items = await asyncio.to_thread(client.search_album, keyword, int(page))
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("搜索失败：%s", exc, exc_info=True)
            return {"success": False, "content": f"搜索失败：{exc}"}

        if not items:
            return {"success": True, "content": f"没有找到与「{keyword}」相关的漫画。"}

        lines = [f"「{keyword}」的搜索结果（第 {page} 页）："]
        for item in items:
            tags = "、".join(item["tags"][:6]) or "无"
            lines.append(f"[JM{item['id']}] {item['title']}\n  标签：{tags}")
        return {"success": True, "content": "\n".join(lines)}

    @Tool(
        "jmcomic_get_album_detail",
        brief_description="查询禁漫天堂漫画的详情",
        detailed_description=(
            "根据漫画 ID 查询详情，返回标题、作者、标签、观看 / 点赞、章节列表等信息。"
            "使用场景：用户想知道某部漫画的具体信息或章节。参数 album_id 为必填漫画 ID。"
        ),
        parameters=[
            ToolParameterInfo(
                name="album_id",
                param_type=ToolParamType.STRING,
                description="漫画 ID（车号），如 123456",
                required=True,
            ),
        ],
    )
    async def get_album_detail_tool(self, album_id: str, **kwargs: Any) -> dict[str, Any]:
        """查询漫画详情（供 LLM 调用）。"""
        del kwargs
        if not self.config.plugin.enabled:
            return {"success": False, "content": "插件未启用。"}

        album_id = (album_id or "").strip()
        if not album_id.isdigit():
            return {"success": False, "content": "漫画 ID 必须是纯数字。"}

        try:
            client = self._get_client()
            album = await asyncio.to_thread(client.get_album_detail, album_id)
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.error("获取详情失败：%s", exc, exc_info=True)
            return {"success": False, "content": f"获取详情失败：{exc}"}

        lines = [
            f"📖 标题：{album['title']}",
            f"🆔 车号：JM{album['id']}",
            f"✍️ 作者：{'、'.join(album['authors']) or '未知'}",
            f"🏷️ 标签：{'、'.join(album['tags']) or '无'}",
            f"📅 上架日期：{album['pub_date'] or '未知'}",
            f"📄 总页数：{album['page_count']}",
            f"👀 观看：{album['views']}  ❤️ 点赞：{album['likes']}  "
            f"💬 评论：{album['comment_count']}",
        ]
        if album["works"]:
            lines.append(f"📚 作品：{'、'.join(album['works'])}")
        if album["actors"]:
            lines.append(f"🎭 人物：{'、'.join(album['actors'])}")
        lines.append(f"📑 章节（共 {len(album['episodes'])}）：")
        for ep in album["episodes"][:20]:
            lines.append(f"  第{ep['index']}话 {ep['title']}（id: {ep['id']}）")
        if len(album["episodes"]) > 20:
            lines.append(f"  ... 仅显示前 20 章，共 {len(album['episodes'])} 章")
        return {"success": True, "content": "\n".join(lines)}


def create_plugin() -> JmcomicPlugin:
    """创建插件实例（Runner 入口）。"""
    return JmcomicPlugin()
