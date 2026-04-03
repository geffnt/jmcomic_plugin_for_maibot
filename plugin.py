
# pyright: reportIncompatibleVariableOverride=false
# pyright: reportIncompatibleMethodOverride=false
# pyright: reportMissingImports=false
# pyright: reportMissingTypeArgument=false

"""
JMComic-Plugin-for-MaiBot 插件

把 JMComic-Crawler-Python (https://github.com/hect0x7/JMComic-Crawler-Python) 接入麦麦。
支持漫画搜索、章节下载、图片发送等功能。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import tempfile
import time
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
import base64

from src.common.logger import get_logger
from src.plugin_system import (
    ActionInfo,
    BaseAction,
    BaseCommand,
    BaseEventHandler,
    BasePlugin,
    BaseTool,
    CommandInfo,
    ConfigField,
    EventHandlerInfo,
    ReplyContentType,
    ToolInfo,
    register_plugin,
)

if TYPE_CHECKING:
    from src.chat.message_receive.message import MessageRecv

logger = get_logger("jmcomic_plugin")

# 依赖安装状态追踪
_dependencies_installed = {
    "jmcomic": False,
    "reportlab": False,
    "PIL": False
}

def install_package(package_name: str) -> bool:
    """使用 pip 安装指定的包。
    
    参数：
        package_name: 要安装的包名
        
    返回：
        安装成功返回 True，失败返回 False
    """
    try:
        logger.info(f"正在安装依赖包: {package_name}")
        # 使用当前 Python 解释器的 pip 安装
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"{package_name} 安装成功")
        logger.debug(f"安装输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{package_name} 安装失败: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"安装 {package_name} 时出错: {e}")
        return False

def ensure_dependencies(packages: List[str]) -> bool:
    """确保所有依赖包都已安装，未安装则自动安装。
    
    参数：
        packages: 包名列表（注意：PIL对应的安装包名是Pillow）
        
    返回：
        所有包都可用返回 True，否则返回 False
    """
    all_available = True
    
    for package in packages:
        # 处理特殊情况：PIL的安装包名是Pillow
        install_name = "Pillow" if package == "PIL" else package
        
        if _dependencies_installed.get(package, False):
            continue
            
        try:
            __import__(package)
            _dependencies_installed[package] = True
            logger.debug(f"{package} 已安装")
        except ImportError:
            logger.warning(f"{package} 未安装，尝试自动安装...")
            if install_package(install_name):
                try:
                    __import__(package)
                    _dependencies_installed[package] = True
                    logger.info(f"{package} 安装并导入成功")
                except ImportError:
                    logger.error(f"{package} 安装后仍无法导入")
                    all_available = False
            else:
                all_available = False
    
    return all_available


# ============================================================
# 工具：跳过 VLM 识图的上下文管理器
# ============================================================
@contextlib.asynccontextmanager
async def skip_vlm_for_images():
    """临时跳过发送图片时的 VLM 识图步骤。"""
    from src.chat.message_receive.message import MessageSending

    original_method = MessageSending._process_single_segment

    async def _patched_process_single_segment(self, segment):
        """对 image 类型直接返回占位符，其余走原逻辑。"""
        if segment.type == "image":
            return "[图片]"
        return await original_method(self, segment)

    MessageSending._process_single_segment = _patched_process_single_segment
    try:
        yield
    finally:
        MessageSending._process_single_segment = original_method


# ============================================================
# 工具：创建 PDF 文件
# ============================================================
def create_pdf_from_images(image_files: list, output_path: Path, album_title: str = "",
                          max_resolution: int = 1500, jpeg_quality: int = 75) -> bool:
    """将图片列表合并为 PDF 文件，并进行压缩以减小文件大小。
    
    参数：
        image_files: 图片文件路径列表
        output_path: 输出 PDF 文件路径
        album_title: 专辑标题（用于 PDF 元数据）
        max_resolution: 图片最大分辨率（长边像素数，默认1500）
        jpeg_quality: JPEG压缩质量（1-95，默认75，推荐65-85）
    
    返回：
        成功返回 True，失败返回 False
    """
    try:
        # 确保 reportlab 和 PIL (Pillow) 已安装
        if not ensure_dependencies(["reportlab", "PIL"]):
            logger.error("无法安装 PDF 生成所需的依赖包")
            return False
            
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm, mm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
        from PIL import Image
        import io
        
        # 创建 PDF
        c = canvas.Canvas(str(output_path), pagesize=A4)
        
        total_original_size = 0
        total_compressed_size = 0
        
        for i, img_path in enumerate(image_files):
            try:
                # 获取原始文件大小
                original_size = img_path.stat().st_size
                total_original_size += original_size
                
                # 打开图片
                img = Image.open(img_path)
                
                # 转换为RGB模式（处理RGBA等模式）
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if img.mode in ('RGBA', 'LA'):
                        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = background
                    else:
                        img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_width, img_height = img.size
                
                # 压缩图片：限制最大分辨率
                max_dim = max(img_width, img_height)
                if max_dim > max_resolution:
                    scale_factor = max_resolution / max_dim
                    new_width = int(img_width * scale_factor)
                    new_height = int(img_height * scale_factor)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    img_width, img_height = new_width, new_height
                    logger.debug(f"图片已缩放: {img_path.name} -> {img_width}x{img_height}")
                
                # 将图片压缩为JPEG格式到内存
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=jpeg_quality, optimize=True)
                img_buffer.seek(0)
                compressed_size = len(img_buffer.getvalue())
                total_compressed_size += compressed_size
                
                # 计算图片在 A4 纸上的位置（居中）
                page_width, page_height = A4
                margin = 1 * cm
                
                # 计算缩放比例
                available_width = page_width - 2 * margin
                available_height = page_height - 2 * margin
                
                scale_w = available_width / img_width
                scale_h = available_height / img_height
                scale = min(scale_w, scale_h)
                
                # 计算实际尺寸和位置
                scaled_width = img_width * scale
                scaled_height = img_height * scale
                x = (page_width - scaled_width) / 2
                y = (page_height - scaled_height) / 2
                
                # 如果不是第一页，创建新页面
                if i > 0:
                    c.showPage()
                
                # 使用压缩后的图片数据绘制
                img_reader = ImageReader(img_buffer)
                c.drawImage(img_reader, x, y, width=scaled_width, height=scaled_height)
                
                # 计算压缩率
                compression_ratio = (original_size - compressed_size) / original_size * 100 if original_size > 0 else 0
                logger.info(f"已添加第 {i+1}/{len(image_files)} 张图片到 PDF (压缩: {compression_ratio:.1f}%)")
                
            except Exception as e:
                logger.warning(f"处理图片 {img_path} 失败: {e}")
                continue
        
        # 保存 PDF
        c.save()
        
        # 计算总体压缩效果
        pdf_size = output_path.stat().st_size
        original_total_mb = total_original_size / (1024 * 1024)
        pdf_total_mb = pdf_size / (1024 * 1024)
        logger.info(f"PDF 创建成功: {output_path} (原始: {original_total_mb:.2f}MB, PDF: {pdf_total_mb:.2f}MB)")
        return True
        
    except ImportError:
        logger.error("缺少 reportlab 库，请运行: pip install reportlab")
        return False
    except Exception as e:
        logger.error(f"创建 PDF 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ============================================================
# 工具：创建加密 ZIP 文件
# ============================================================
def create_encrypted_zip_from_images(image_files: list, output_path: Path, password: str = "maimai", album_title: str = "") -> bool:
    """将图片列表打包为加密的 ZIP 文件。
    
    参数：
        image_files: 图片文件路径列表
        output_path: 输出 ZIP 文件路径
        password: ZIP 加密密码
        album_title: 专辑标题（用于 ZIP 注释）
    
    返回：
        成功返回 True，失败返回 False
    """
    try:
        import zipfile
        from PIL import Image
        import io
        
        # 创建加密 ZIP 文件
        with zipfile.ZipFile(
            output_path,
            'w',
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True
        ) as zf:
            # 设置 ZIP 注释
            if album_title:
                zf.comment = f"JMComic - {album_title}".encode('utf-8')
            
            for i, img_path in enumerate(image_files):
                try:
                    # 打开图片并转换为 PNG 格式
                    img = Image.open(img_path)
                    
                    # 如果图片有调色板模式，转换为 RGB
                    if img.mode in ('P', 'RGBA'):
                        img = img.convert('RGBA')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 将图片保存到内存中的 PNG 格式
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    # 生成文件名（使用序号）
                    filename = f"{i + 1:04d}.png"
                    
                    # 添加到 ZIP 文件（使用密码加密）
                    zf.writestr(
                        filename,
                        img_buffer.getvalue(),
                        compress_type=zipfile.ZIP_DEFLATED
                    )
                    
                    logger.info(f"已添加第 {i+1}/{len(image_files)} 张图片到 ZIP")
                    
                except Exception as e:
                    logger.warning(f"处理图片 {img_path} 失败: {e}")
                    continue
        
        logger.info(f"ZIP 创建成功: {output_path}")
        
        # 使用 pyminizip 进行 AES 加密（如果可用）
        try:
            import pyminizip
            # 创建临时未加密的 ZIP
            temp_zip = output_path.with_suffix(".temp.zip")
            output_path.rename(temp_zip)
            
            # 使用 pyminizip 加密
            pyminizip.compress(
                str(temp_zip),
                None,
                str(output_path),
                password,
                5  # 压缩级别
            )
            
            # 删除临时文件
            temp_zip.unlink()
            logger.info(f"ZIP 加密成功: {output_path}")
            
        except ImportError:
            # pyminizip 不可用，使用标准库的简单加密
            logger.warning("pyminizip 不可用，使用标准 ZIP 加密（较弱）")
            # Python 标准库的 zipfile 不支持 AES 加密
            # 我们使用 pyzipper 作为替代
            try:
                import pyzipper
                
                # 重新创建 ZIP 文件，使用 AES 加密
                temp_zip = output_path.with_suffix(".temp.zip")
                output_path.rename(temp_zip)
                
                with pyzipper.AESZipFile(
                    output_path,
                    'w',
                    compression=pyzipper.ZIP_DEFLATED,
                    encryption=pyzipper.WZ_AES
                ) as zf:
                    zf.setpassword(password.encode('utf-8'))
                    
                    if album_title:
                        zf.comment = f"JMComic - {album_title}".encode('utf-8')
                    
                    for i, img_path in enumerate(image_files):
                        try:
                            img = Image.open(img_path)
                            if img.mode in ('P', 'RGBA'):
                                img = img.convert('RGBA')
                            elif img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            img_buffer = io.BytesIO()
                            img.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            filename = f"{i + 1:04d}.png"
                            zf.writestr(filename, img_buffer.getvalue())
                            
                        except Exception as e:
                            logger.warning(f"处理图片 {img_path} 失败: {e}")
                            continue
                
                temp_zip.unlink()
                logger.info(f"ZIP AES 加密成功: {output_path}")
                
            except ImportError:
                logger.warning("pyzipper 不可用，ZIP 文件未加密")
                # 如果没有加密库，至少创建未加密的 ZIP
        
        return True
        
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        return False
    except Exception as e:
        logger.error(f"创建 ZIP 失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


# ============================================================
# 工具：classproperty 描述符
# ============================================================
class classproperty(property):
    """支持类访问与实例访问的属性描述符。"""

    def __get__(self, obj: Any, owner: Optional[type] = None) -> Any:
        owner_type: type = owner or type(obj)
        fget_func = self.fget
        if fget_func is None:
            raise AttributeError("unreadable classproperty")
        return fget_func(owner_type)


# ============================================================
# JMComic 客户端封装
# ============================================================
class JMComicClient:
    """JMComic-Crawler-Python 封装客户端。"""

    def __init__(
        self,
        base_dir: str,
        domain: Optional[str] = None,
        proxy: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        初始化 JMComic 客户端。

        参数：
            base_dir: 下载目录
            domain: JMComic 域名
            proxy: 代理地址
            username: 登录用户名
            password: 登录密码
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.proxy = proxy
        self.username = username
        self.password = password
        self._jmc = None
        self._initialized = False
        self._client = None  # JmApiClient 实例

    def _init_jmcomic(self) -> None:
        """初始化 jmcomic 库（延迟加载）。"""
        if self._initialized:
            return

        try:
            # 确保 jmcomic 已安装
            if not ensure_dependencies(["jmcomic"]):
                raise RuntimeError("无法安装 jmcomic 依赖包")
                
            import jmcomic
            from jmcomic import JmApiClient, JmModuleConfig
            
            self._jmc = jmcomic

            # 创建 JmApiClient 实例
            postman = JmModuleConfig.new_postman()
            domain_list = JmModuleConfig.DOMAIN_API_LIST
            self._client = JmApiClient(postman, domain_list)

            self._initialized = True
            
            logger.info("JMComic 库初始化成功")
            
        except Exception as e:
            raise RuntimeError(f"JMComic 初始化失败: {e}")

    async def search_album(self, keyword: str, page: int = 1) -> List[Dict[str, Any]]:
        """搜索漫画。"""
        return await asyncio.to_thread(self._search_album_sync, keyword, page)

    def _search_album_sync(self, keyword: str, page: int) -> List[Dict[str, Any]]:
        self._init_jmcomic()
        try:
            logger.info(f"使用 JmApiClient 进行搜索，关键字: {keyword}，页码: {page}")
            
            # 使用 search_site 方法搜索（最简单的搜索方式）
            search_result = self._client.search_site(keyword, page)
            
            albums = []
            for album in search_result.album_list:
                albums.append({
                    "id": getattr(album, "album_id", getattr(album, "id", "")),
                    "title": getattr(album, "name", getattr(album, "title", "")),
                    "author": getattr(album, "author", "未知"),
                    "description": getattr(album, "description", ""),
                })
                
            logger.info(f"搜索成功，找到 {len(albums)} 本漫画")
            return albums
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return []

    async def get_album_info(self, album_id: str) -> Optional[Dict[str, Any]]:
        """获取漫画详情。"""
        return await asyncio.to_thread(self._get_album_info_sync, album_id)

    def _get_album_info_sync(self, album_id: str) -> Optional[Dict[str, Any]]:
        self._init_jmcomic()
        try:
            logger.info(f"尝试获取专辑信息: {album_id}")
            
            # 直接使用 JmApiClient 获取专辑信息
            album_detail = self._client.get_album_detail(album_id)
            logger.info(f"成功获取专辑对象: {type(album_detail)}")
            logger.info(f"专辑对象属性: {dir(album_detail)}")
            
            # 提取章节信息
            chapters = []
            
            # 尝试多种方式获取章节
            episodes = []
            if hasattr(album_detail, 'episode_list'):
                episodes = album_detail.episode_list
                logger.info(f"找到 {len(episodes)} 个章节 (通过 episode_list 属性)")
            elif hasattr(album_detail, 'episodes'):
                episodes = album_detail.episodes
                logger.info(f"找到 {len(episodes)} 个章节 (通过 episodes 属性)")
            elif hasattr(album_detail, 'chapter_list'):
                episodes = album_detail.chapter_list
                logger.info(f"找到 {len(episodes)} 个章节 (通过 chapter_list 属性)")
            elif hasattr(album_detail, 'photos'):
                episodes = album_detail.photos
                logger.info(f"找到 {len(episodes)} 个章节 (通过 photos 属性)")
            else:
                logger.warning("未找到章节列表属性")
            
            for i, episode in enumerate(episodes):
                # 尝试多种方式获取章节 ID
                photo_id = None
                for attr in ['photo_id', 'id', 'chapter_id']:
                    if hasattr(episode, attr):
                        photo_id = str(getattr(episode, attr))
                        break
                
                if photo_id is None:
                    photo_id = str(i + 1)
                
                # 尝试多种方式获取章节标题
                title = None
                for attr in ['name', 'title', 'chapter_name']:
                    if hasattr(episode, attr):
                        title = getattr(episode, attr)
                        break
                
                if title is None:
                    title = f"第 {i + 1} 章"
                
                chapters.append({
                    "index": i + 1,
                    "id": photo_id,
                    "photo_id": photo_id,
                    "title": title
                })
            
            # 获取专辑标题
            title = None
            for attr in ['name', 'title', 'album_name']:
                if hasattr(album_detail, attr):
                    title = getattr(album_detail, attr)
                    break
            
            if title is None:
                title = f"JMComic 专辑 {album_id}"
            
            # 获取作者
            author = None
            for attr in ['author', 'author_name']:
                if hasattr(album_detail, attr):
                    author = getattr(album_detail, attr)
                    break
            
            if author is None:
                author = "未知"
            
            # 获取描述
            description = None
            for attr in ['description', 'intro', 'introduction']:
                if hasattr(album_detail, attr):
                    description = getattr(album_detail, attr)
                    break
            
            if description is None:
                description = ""
            
            logger.info(f"专辑信息 - 标题: {title}, 作者: {author}, 章节数: {len(chapters)}")
            
            # 如果没有找到章节，但可能需要尝试其他方法
            if len(chapters) == 0:
                logger.warning("未找到章节，尝试使用备用方法")
                # 至少返回一个默认章节，让下载可以继续
                chapters.append({
                    "index": 1,
                    "id": str(album_id),
                    "photo_id": str(album_id),
                    "title": "第 1 章"
                })
            
            return {
                "id": str(album_id),
                "title": title,
                "author": author,
                "description": description,
                "chapters": chapters
            }
            
        except Exception as e:
            logger.error(f"获取漫画详情失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def download_chapter(self, album_id: str, chapter_index: int) -> Optional[Path]:
        """下载章节并返回目录路径。"""
        return await asyncio.to_thread(self._download_chapter_sync, album_id, chapter_index)

    def _download_chapter_sync(self, album_id: str, chapter_index: int) -> Optional[Path]:
        self._init_jmcomic()
        try:
            logger.info(f"开始下载漫画 {album_id} 第 {chapter_index} 章")
            
            # 获取专辑信息以获取章节的 photo_id
            album_info = self._get_album_info_sync(album_id)
            photo_id = None
            
            if album_info and 'chapters' in album_info and chapter_index >= 1 and chapter_index <= len(album_info['chapters']):
                # 获取章节的 photo_id
                chapter_info = album_info['chapters'][chapter_index - 1]
                photo_id = chapter_info['photo_id']
                logger.info(f"从专辑信息获取到 photo_id: {photo_id}")
            else:
                logger.warning(f"无法从专辑信息获取章节，尝试直接使用 album_id 作为 photo_id")
                # 如果无法获取章节信息，尝试直接使用 album_id（有些漫画可能只有一章）
                photo_id = album_id
            
            # 创建下载目录 - 直接在 jmcomic 文件夹下创建
            chapter_dir = self.base_dir / f"{album_id}_{chapter_index}"
            chapter_dir.mkdir(parents=True, exist_ok=True)
    
            # 使用 jmcomic 模块下载章节
            import jmcomic
            import os
    
            # 切换工作目录到 jmcomic 文件夹，让 jmcomic 下载到这里
            original_dir = os.getcwd()
            os.chdir(str(self.base_dir))
            logger.info(f"切换工作目录到: {self.base_dir}")
    
            try:
                # 尝试使用简单的下载方法
                try:
                    # 方法1: 尝试使用 download_photo
                    logger.info(f"尝试下载 photo_id: {photo_id}")
                    photo = jmcomic.download_photo(photo_id)
                    logger.info(f"下载完成，检查文件...")
    
                except Exception as e1:
                    logger.warning(f"方法1失败: {e1}")
                    # 方法2: 尝试下载整个专辑
                    try:
                        logger.info(f"尝试下载整个专辑: {album_id}")
                        jmcomic.download_album(album_id)
                    except Exception as e2:
                        logger.warning(f"方法2也失败: {e2}")
                        # 如果都失败，抛出异常
                        raise
            finally:
                # 恢复原始工作目录
                os.chdir(original_dir)
                logger.info(f"恢复工作目录到: {original_dir}")
            
            # 检查下载是否成功 - 搜索所有可能的下载位置
            logger.info(f"搜索下载的文件...")
            
            # 检查我们创建的目录
            if chapter_dir.exists() and any(chapter_dir.iterdir()):
                logger.info(f"在预期位置找到下载内容: {chapter_dir}")
                return chapter_dir
            
            # 检查 base_dir 目录下的所有子文件夹（jmcomic 文件夹内）
            if self.base_dir.exists():
                for item in self.base_dir.iterdir():
                    if item.is_dir():
                        image_files = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp')]
                        if image_files:
                            logger.info(f"在 jmcomic 文件夹找到下载内容: {item}，图片数量: {len(image_files)}")
                            return item
    
            # 如果 base_dir 下没有找到，检查 base_dir 本身是否有图片
            if self.base_dir.exists():
                image_files = [f for f in self.base_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp')]
                if image_files:
                    logger.info(f"在 jmcomic 文件夹根目录找到图片: {len(image_files)} 张")
                    return self.base_dir
            
            logger.error(f"无法找到与漫画 {album_id} 相关的下载文件")
            return None
                
        except Exception as e:
            logger.error(f"下载章节失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# ============================================================
# 命令组件
# ============================================================
class JMComicCommand(BaseCommand):
    """JMComic 漫画命令。"""

    command_name: str = "jm漫画"
    command_description: str = "从 JMComic 搜索和下载漫画"
    command_pattern: str = r"^(?P<trigger>/jm|jm漫画|禁漫)\s*(?P<args>.*)?$"
    command_help: str = (
        "/jm <漫画ID> [章节号] [pdf/jpg] - 下载并发送漫画\n"
        "示例：\n"
        "  /jm 123456 - 下载漫画 123456 的第 1 章（默认PDF格式）\n"
        "  /jm 123456 2 - 下载漫画 123456 的第 2 章\n"
        "  /jm 123456 pdf - 以PDF格式发送\n"
        "  /jm 123456 jpg - 以图片格式发送\n"
        "  /jm 123456 2 jpg - 下载第2章并以图片格式发送\n"
    )
    command_examples: List[str] = [
        "/jm 123456",
        "/jm 123456 2",
        "/jm 123456 pdf",
        "/jm 123456 jpg",
    ]
    enable_command: bool = True

    def __init__(self, message: MessageRecv, plugin_config: Optional[Dict[str, Any]] = None):
        super().__init__(message, plugin_config)
        self._cooldown_cache: Dict[str, float] = {}

    async def execute(self) -> Tuple[bool, Optional[str], int]:
        """命令执行入口。"""
        args_str: str = str(self.matched_groups.get("args", "") or "").strip()

        if args_str.lower() in ("help", "帮助", "?", "？", ""):
            await self.send_text(self.command_help)
            return (True, "显示帮助", 2)

        # 解析参数
        args = args_str.split()
        if len(args) < 1:
            await self.send_text("请输入漫画ID，格式：/jm <漫画ID> [章节号] [pdf/jpg]")
            return (True, "显示帮助", 2)
            
        album_id = args[0]
        chapter_index = 1
        output_format = None  # None 表示使用配置默认值
        
        # 解析其他参数
        for arg in args[1:]:
            arg_lower = arg.lower()
            if arg_lower in ("pdf", "jpg", "jpeg", "png"):
                output_format = arg_lower
            elif arg.isdigit():
                chapter_index = int(arg)
            
        # 获取插件目录，在插件文件夹内创建 jmcomic 文件夹
        plugin_dir = Path(__file__).parent  # sigef_jmcomic-plugin 目录
        base_dir: str = str(plugin_dir / "jmcomic")
        domain: Optional[str] = self.get_config("api.domain")
        proxy: Optional[str] = self.get_config("api.proxy")
        username: Optional[str] = self.get_config("auth.username")
        password: Optional[str] = self.get_config("auth.password")
        cooldown: int = self._safe_int(self.get_config("features.cooldown_seconds", 30), 30)
        max_pages_per_send: int = self._safe_int(self.get_config("features.max_pages_per_send", 1000), 1000)
        use_forward: bool = self._safe_bool(self.get_config("features.use_forward_message", True))
        
        # 根据命令参数或配置决定输出格式，默认PDF
        if output_format == "pdf":
            send_as_pdf = True
        elif output_format in ("jpg", "jpeg", "png"):
            send_as_pdf = False
        else:
            # 使用配置默认值，默认为PDF
            send_as_pdf = self._safe_bool(self.get_config("features.send_as_pdf", True))
        # 获取每个PDF的图片数量，限制在0-60之间
        images_per_pdf: int = self._safe_int(self.get_config("features.images_per_pdf", 40), 40)
        images_per_pdf = max(0, min(60, images_per_pdf))  # 限制在0-60之间

        user_id: str = self._safe_user_id()
        now: float = time.time()
        last_use: float = self._cooldown_cache.get(user_id, 0.0)
        if cooldown > 0 and (now - last_use) < cooldown:
            remaining: int = int(cooldown - (now - last_use)) + 1
            await self.send_text(f"冷却中，请 {remaining} 秒后再试~")
            return (True, "冷却中", 2)

        client: JMComicClient = JMComicClient(
            base_dir=base_dir,
            domain=domain,
            proxy=proxy,
            username=username,
            password=password,
        )

        # 直接处理下载
        if not album_id or not album_id.isdigit():
            await self.send_text("请输入有效的漫画 ID，如：/jm 123456")
            return (True, "无效 ID", 2)
            
        chapter_dir = None
        try:
            await self.send_text(f"正在下载漫画 {album_id} 第 {chapter_index} 章...")
            chapter_dir = await client.download_chapter(album_id, chapter_index)

            if not chapter_dir:
                await self.send_text("下载失败，请检查漫画 ID 和章节号是否正确")
                return (True, "下载失败", 2)

            await self._send_chapter_images(chapter_dir, max_pages_per_send, use_forward, send_as_pdf, album_id, chapter_index, images_per_pdf)
            
            self._cooldown_cache[user_id] = time.time()
            return (True, "下载并发送成功", 2)
            
        finally:
            # 程序结束时清空整个 jmcomic 文件夹
            import shutil
            try:
                plugin_dir = Path(__file__).parent
                jmcomic_dir = plugin_dir / "jmcomic"
                if jmcomic_dir.exists():
                    shutil.rmtree(jmcomic_dir)
                    logger.info(f"已清空 jmcomic 文件夹: {jmcomic_dir}")
            except Exception as e:
                logger.warning(f"清空 jmcomic 文件夹失败: {e}")


    async def _send_chapter_images(
        self,
        chapter_dir: Path,
        max_pages: int,
        use_forward: bool,
        send_as_pdf: bool,
        album_id: str,
        chapter_index: int,
        images_per_pdf: int = 40,
    ) -> None:
        """发送章节图片。
        
        参数：
            chapter_dir: 章节目录
            max_pages: 最大发送页数
            use_forward: 是否使用合并转发
            send_as_pdf: 是否发送为PDF
            album_id: 专辑ID
            chapter_index: 章节索引
            images_per_pdf: 每个PDF包含的图片数量（0表示不限制，最大60）
        """
        image_files = sorted([
            f for f in chapter_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.gif', '.webp')
        ])[:max_pages]

        if not image_files:
            await self.send_text("章节中没有找到图片")
            return

        # 如果选择 PDF 发送
        if send_as_pdf:
            # 使用配置的每个PDF图片数量，限制在0-60之间
            # 0表示不限制（所有图片放入一个PDF）
            max_images_per_pdf = max(0, min(60, images_per_pdf))
            total_images = len(image_files)
            
            # 计算需要分多少个PDF
            if max_images_per_pdf == 0:
                # 不限制，所有图片放入一个PDF
                num_pdfs = 1
                MAX_IMAGES_PER_PDF = total_images
            elif total_images < 60:
                # 如果图片总数小于60张，即使配置更小，也合并成一个PDF
                num_pdfs = 1
                MAX_IMAGES_PER_PDF = total_images
            else:
                MAX_IMAGES_PER_PDF = max_images_per_pdf
                num_pdfs = (total_images + MAX_IMAGES_PER_PDF - 1) // MAX_IMAGES_PER_PDF
            
            if num_pdfs > 1:
                await self.send_text(f"共 {total_images} 张图片，将分为 {num_pdfs} 个 PDF 发送（每个最多 {MAX_IMAGES_PER_PDF} 张）")
            else:
                await self.send_text(f"准备将 {total_images} 张图片合并为 PDF 发送...")
            
            pdf_paths_to_delete = []
            send_success = True
            
            for pdf_index in range(num_pdfs):
                start_idx = pdf_index * MAX_IMAGES_PER_PDF
                end_idx = min((pdf_index + 1) * MAX_IMAGES_PER_PDF, total_images)
                batch_images = image_files[start_idx:end_idx]
                
                # 生成PDF文件名
                if num_pdfs > 1:
                    pdf_name = f"JM{album_id}_第{chapter_index}章_第{pdf_index + 1}部分"
                else:
                    pdf_name = f"JM{album_id}_第{chapter_index}章"
                
                pdf_path = chapter_dir / f"{pdf_name}.pdf"
                pdf_paths_to_delete.append(pdf_path)
                
                if num_pdfs > 1:
                    await self.send_text(f"正在生成第 {pdf_index + 1}/{num_pdfs} 个 PDF ({len(batch_images)} 张图片)...")
                
                if create_pdf_from_images(batch_images, pdf_path, f"JM{album_id}"):
                    # 发送 PDF 文件
                    try:
                        # 读取 PDF 文件为 base64
                        pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
                        # 文件消息需要包含文件名和文件内容
                        file_data = {
                            "file": f"base64://{pdf_b64}",
                            "name": f"{pdf_name}.pdf"
                        }
                        await self.send_custom(
                            message_type="file",
                            content=file_data,
                            display_message=f"{pdf_name}.pdf"
                        )
                        logger.info(f"PDF 文件已发送: {pdf_path}")
                    except Exception as e:
                        logger.error(f"发送 PDF 失败: {e}")
                        send_success = False
                        # 回退到图片发送
                        await self.send_text(f"PDF 发送失败，将以图片形式发送...")
                        if use_forward:
                            await self._send_as_forward(image_files, album_id, chapter_index)
                        else:
                            await self._send_as_separate(image_files, album_id, chapter_index)
                        break
                else:
                    await self.send_text("创建 PDF 失败，将以图片形式发送...")
                    send_success = False
                    # 回退到图片发送
                    if use_forward:
                        await self._send_as_forward(image_files, album_id, chapter_index)
                    else:
                        await self._send_as_separate(image_files, album_id, chapter_index)
                    break
            
            # 删除所有临时 PDF 文件
            for pdf_path in pdf_paths_to_delete:
                try:
                    if pdf_path.exists():
                        pdf_path.unlink()
                        logger.info(f"已删除临时 PDF 文件: {pdf_path}")
                except Exception as e:
                    logger.warning(f"删除临时 PDF 文件失败: {e}")
            
            if send_success and num_pdfs > 1:
                await self.send_text(f"全部 {num_pdfs} 个 PDF 发送完成！")
            
            return

        await self.send_text(f"准备发送 {len(image_files)} 张图片...")

        # 根据 use_forward 参数选择发送方式
        if use_forward:
            # 合并转发发送
            await self._send_as_forward(image_files, album_id, chapter_index)
        else:
            # 逐张发送
            await self._send_as_separate(image_files, album_id, chapter_index)

    async def _send_as_forward(self, image_files: List[Path], album_id: str, chapter_index: int) -> None:
        """以合并转发消息形式发送。"""
        from src.config.config import global_config

        bot_qq: str = str(global_config.bot.qq_account)
        bot_name: str = str(global_config.bot.nickname)

        async def read_image_to_b64(img_path: Path) -> Optional[str]:
            try:
                return base64.b64encode(img_path.read_bytes()).decode("ascii")
            except Exception as e:
                logger.warning(f"读取图片失败: {e}")
                return None

        tasks = [asyncio.create_task(read_image_to_b64(f)) for f in image_files]
        base64_results = await asyncio.gather(*tasks)

        forward_messages: List[Any] = []
        for i, (img_path, img_b64) in enumerate(zip(image_files, base64_results), 1):
            content: List[Any] = [(ReplyContentType.TEXT, f"第 {i} 页")]
            if img_b64:
                content.append((ReplyContentType.IMAGE, img_b64))
            forward_messages.append((bot_qq, bot_name, content))

        async with skip_vlm_for_images():
            await self.send_forward(forward_messages, storage_message=True)

    async def _send_as_separate(self, image_files: List[Path], album_id: str, chapter_index: int) -> None:
        """逐条发送图片。"""
        async with skip_vlm_for_images():
            for i, img_path in enumerate(image_files, 1):
                try:
                    img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                    await self.send_text(f"第 {i} 页")
                    await self.send_image(img_b64)
                except Exception as e:
                    logger.warning(f"发送图片失败: {e}")

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        """安全转换为 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_bool(value: Any) -> bool:
        """安全转换为 bool。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    def _safe_user_id(self) -> str:
        """安全获取用户 ID。"""
        try:
            if (
                hasattr(self.message, "message_info")
                and hasattr(self.message.message_info, "user_info")
                and self.message.message_info.user_info is not None
                and hasattr(self.message.message_info.user_info, "user_id")
            ):
                return str(self.message.message_info.user_info.user_id)
        except Exception:
            pass
        return "unknown"


# ============================================================
# 插件主类
# ============================================================
@register_plugin
class JMComicPlugin(BasePlugin):
    """JMComic-Plugin-for-MaiBot 插件。"""

    plugin_name: str = "jmcomic_plugin"
    enable_plugin: bool = True
    dependencies: list[str] = []
    python_dependencies: list[str] = ["jmcomic"]
    config_file_name: str = "config.toml"

    @classproperty
    def config_schema(cls) -> Dict[str, Any]:
        """插件配置结构。"""
        return {
            "plugin": {
                "enabled": ConfigField(
                    type=bool,
                    default=False,
                    description="是否启用插件",
                ),
                "config_version": ConfigField(
                    type=str,
                    default="1.0.0",
                    description="配置版本号",
                ),
            },
            "components": {
                "enable_command": ConfigField(
                    type=bool,
                    default=True,
                    description="是否启用 JMComic 命令",
                ),
            },
            "api": {
                "domain": ConfigField(
                    type=str,
                    default="",
                    description="JMComic 域名",
                ),
                "proxy": ConfigField(
                    type=str,
                    default="",
                    description="代理地址",
                ),
            },
            "download": {
                "base_dir": ConfigField(
                    type=str,
                    default="./jmcomic_downloads",
                    description="漫画下载目录",
                ),
            },
            "auth": {
                "username": ConfigField(
                    type=str,
                    default="",
                    description="JMComic 用户名",
                ),
                "password": ConfigField(
                    type=str,
                    default="",
                    description="JMComic 密码",
                ),
            },
            "features": {
                "cooldown_seconds": ConfigField(
                    type=int,
                    default=30,
                    description="命令冷却时间",
                ),
                "max_pages_per_send": ConfigField(
                    type=int,
                    default=1000,
                    description="单次最多发送页数",
                ),
                "use_forward_message": ConfigField(
                    type=bool,
                    default=True,
                    description="使用合并转发",
                ),
                "send_as_pdf": ConfigField(
                    type=bool,
                    default=False,
                    description="将图片合并为 PDF 发送",
                ),
                "images_per_pdf": ConfigField(
                    type=int,
                    default=40,
                    description="每个PDF包含的图片数量（0表示不限制，最大60）",
                ),
            },
        }

    def get_plugin_components(self):
        """返回插件组件列表。"""
        components = []
        enable_command = self.get_config("components.enable_command", True)
        if enable_command:
            components.append((JMComicCommand.get_command_info(), JMComicCommand))
        return components
