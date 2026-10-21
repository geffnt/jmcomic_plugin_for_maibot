# JMComic 漫画插件 for MaiBot

基于 [JMComic-Crawler-Python](https://github.com/hect0x7/JMComic-Crawler-Python) 的禁漫天堂漫画插件，支持漫画搜索、详情查询、章节下载，以及 PDF / 图片两种方式发送。

本插件已升级到 **MaiBot 新插件系统（maibot-plugin-sdk）**：使用 `plugin.py` 入口 + `_manifest.json` 元信息，实现 `on_load` / `on_unload` / `on_config_update` / `create_plugin`，能力通过 `@Command` 与 `@Tool` 声明，不使用已废弃的 `@Action`。

## ⚠️ 重要警示

- **群聊环境中发送大量图片有极高封号风险**，强烈建议仅在**私聊**中使用图片发送，或使用默认的 **PDF 模式**。
- 请遵守禁漫天堂的使用条款与当地法律法规，合理使用冷却时间，避免对服务器造成压力。

## 功能特性

- 🔍 **漫画搜索** - 通过关键字搜索漫画（`@Tool`，供 LLM 主动调用）
- 📋 **详情查询** - 查询漫画标题、作者、标签、章节列表（`@Tool`）
- 📥 **章节下载** - 下载指定漫画的指定章节
- 🖼️ **图片发送** - 支持逐张发送或合并转发（图文混合）
- 📄 **PDF 导出** - 将章节图片合并为 PDF 发送（默认，推荐群聊使用）
- 📦 **PDF 压缩** - 自动压缩图片分辨率（最大 1500px）与质量（JPEG 75%）
- ⚙️ **灵活配置** - 支持自定义域名、代理、账号登录、冷却时间等

## 安装

1. 将本目录放入 MaiBot 的 `plugins/` 下（目录名建议保持为插件 ID `geffnt.jmcomic-plugin`，或任意合法目录名）。
2. 启动 MaiBot，Host 会根据 `_manifest.json` 的 `dependencies` 自动安装依赖（`jmcomic`、`reportlab`、`Pillow`）。
3. 在 WebUI 的「插件管理」中启用本插件（默认关闭）。
4. 首次启用后，Runner 会根据 `config_model` 生成运行时 `config.toml`。

> 手动安装依赖：`pip install jmcomic reportlab Pillow`

## 使用方法

### 命令

```
/jm <漫画ID> [章节号] [pdf/jpg]
/jm help
```

- `<漫画ID>`：必填，禁漫天堂漫画 ID（车号）。
- `[章节号]`：可选，下载第几章，默认第 1 章。
- `[pdf/jpg]`：可选输出格式，默认按配置 `features.send_as_pdf`（默认 PDF）。

示例：

```
/jm 123456          # 第 1 章，PDF
/jm 123456 2        # 第 2 章，PDF
/jm 123456 2 jpg    # 第 2 章，图片
/jm help            # 帮助
```

命令触发词同时支持 `/jm`、`jm漫画`、`禁漫` 前缀。

### 工具（LLM 可主动调用）

- `jmcomic_search_album`：按关键字搜索漫画。
- `jmcomic_get_album_detail`：按 ID 查询漫画详情。

## 配置说明

运行时配置由 Runner 根据 `plugin.py` 中的 `config_model` 生成（`config.toml`），配置项与默认值如下：

| 分组 | 字段 | 默认值 | 说明 |
|------|------|--------|------|
| `plugin` | `enabled` | `false` | 是否启用插件 |
| `plugin` | `config_version` | `"1.0.0"` | 配置版本 |
| `components` | `enable_command` | `true` | 是否启用 `/jm` 命令 |
| `api` | `domain` | `""` | JMComic 自定义 API 域名（留空用默认） |
| `api` | `proxy` | `""` | 代理地址，如 `http://127.0.0.1:7890`（留空用系统代理） |
| `download` | `base_dir` | `""` | 下载目录（留空用插件运行时临时目录） |
| `auth` | `username` | `""` | JMComic 用户名（可选） |
| `auth` | `password` | `""` | JMComic 密码 |
| `features` | `cooldown_seconds` | `30` | 命令冷却时间（秒），0 表示不限制 |
| `features` | `max_pending_jobs` | `3` | 正在发送时最多排队多少个命令，超出会提示队列已满 |
| `features` | `max_pages_per_send` | `1000` | 单次最多发送页数 |
| `features` | `use_forward_message` | `true` | 以合并转发（图文混合）方式发送 |
| `features` | `send_as_pdf` | `true` | 将图片合并为 PDF 发送 |
| `features` | `pdf_fallback_to_images` | `false` | PDF 发送失败时是否回退为逐张发图（默认关闭） |
| `features` | `single_pdf_max_pages` | `60` | 页数不超过该值时合成单个 PDF |
| `features` | `images_per_pdf` | `40` | 超过阈值时每份 PDF 的页数（0 表示不拆分，最大 60） |
| `features` | `pdf_target_mb_per_60_pages` | `10` | 60 页 PDF 的目标体积（MB），按页均摊为压缩预算；0 表示不限制 |
| `features` | `pdf_max_resolution` | `1500` | PDF 内图片长边最大像素 |
| `features` | `pdf_jpeg_quality` | `75` | PDF 内图片的起始 JPEG 质量（不超预算时就用它） |

## 实现要点（改动后需注意）

### PDF 分卷规则

由纯函数 `_resolve_pdf_batch_size(total, single_pdf_max_pages, images_per_pdf)` 决定，
默认 `single_pdf_max_pages = 60`、`images_per_pdf = 40`：

| 章节页数 | 发送结果 |
|---------|---------|
| ≤ 60 | 合成 **1 个** PDF（含全部页） |
| 61 | 2 个 PDF：40 + 21 |
| 100 | 3 个 PDF：40 + 40 + 20 |
| 200 | 5 个 PDF：40 × 5 |

`images_per_pdf = 0` 表示超过阈值也不拆分。
单元测试用 `ast` 抽出该纯函数（无需安装 SDK）覆盖了这些边界。

### 消息流程

发 `/jm` 之后依次出现三条消息：

| 时机 | 文本 |
|------|------|
| 受理后、下载前 | `正在下载《作品名》第X章` |
| 生成 PDF 前 | 分卷时：`共 x 张图片，将分为 y 个 PDF 发送`；单卷时：`共 x 页` |
| 之后 | PDF 文件本身（分卷则多条） |

两个要点：

- **作品名要先解析出来**。`JmcomicClient` 因此拆成
  `resolve_chapter()`（返回 `(作品名, photo_id)`）与 `download_photo()` 两步，
  这样提示能在**慢的下载开始前**发出，且不会重复请求一次详情。
  `download_chapter()` 仍保留为两者的便捷封装。
- **提示只在真的分卷时才报份数**，判据是 `pdf_count > 1`（而不是直接比 60），
  这样改了 `single_pdf_max_pages` / `images_per_pdf` 后行为依然自洽。
  该逻辑在纯函数 `_format_page_notice(total, pdf_count)` 里，已由
  单元测试覆盖（59/60 报页数，61 起报份数）。

### 命令串行化（待处理队列）

**前一个命令的 PDF 完全发送结束之前，后续命令只会排队，不会并发执行。**

实现是「单队列 + 单 worker」：`handle_jm` 不直接干活，只把参数快照成
`JmJob` 放进 `self._job_queue`，由唯一的 `_job_worker` 逐个取出执行。
worker 每次都 `await` 完整个任务才取下一个，串行因此是结构性的，
而不是靠锁或计时器。

两种分支的行为：

| 情况 | 行为 |
|------|------|
| 空闲 | 走原有冷却检查，通过则立即执行 |
| 正在发送 / 已有排队 | **跳过冷却检查直接排队**，回 `已加入待处理队列（前面还有 N 个），等当前任务发完自动开始~` |
| 排队数达到 `max_pending_jobs` | 回 `待处理队列已满（N 个），请等当前任务发完后再试~` |

冷却检查刻意放在排队分支**之后**：否则冷却会把本该排队的命令直接挡掉，
就回到了「发送期间命令被丢弃」的老问题。

**一个必须留意的坑：延迟清理不能 await 在任务里。**
`_PDF_KEEP_SECONDS` 是 300 秒，原来写在 `_run_download_job` 的 `finally` 里——
若保持那样，队列里下一个命令要白等 5 分钟。所以清理已挪到独立后台任务
（`_schedule_pdf_cleanup` → `_cleanup_pdfs_later`），
`_run_download_job` 返回即代表本次 PDF 已全部发完。
集成测试把保留期故意设成 10 秒来验证这一点。

### 禁漫接口的 UTF-8 BOM（服务端变更兼容）

**现象**：所有 `/jm` 命令突然全部失败，日志里是一长串

```
请求重试全部失败: [/setting], ['www.cdnhjk.net', 'www.cdngwc.cc', ...]
  1. https://www.cdnhjk.net/setting, retry=0: JmcomicException:
     请求不是json格式，强制重试！响应文本: [{"code":200,"data":"5uESp5Ky…"}]
```

**原因**：2026-10 起，禁漫服务端在响应体前面加了一个 **UTF-8 BOM**
（`b'\xef\xbb\xbf'`）。实测原始字节：

```
content[:24] = b'\xef\xbb\xbf{"code":200,"data":"7'
首个有效字符  = '\ufeff' (码位 65279)
```

而 jmcomic 的报文校验只跳过空格 / 换行 / 制表符，碰到 `\ufeff` 就判成
「不是 json」，于是把每个域名都重试 6 次、全部失败。
注意 `data` 字段本身是 AES 加密的，jmcomic **本来就能解密**——卡住的只是这个首字符检查。
`jmcomic 2.7.7`（2026-09-13）已是 PyPI 最新版，上游尚未处理。

**处理**：插件在 `on_load` 时调 `_install_bom_tolerance()`，把
`AbstractJmClient` / `JmApiClient` / `AsyncJmApiClient` 的响应校验各包一层，
校验前剥掉前导 BOM。只放宽输入、不改变行为，且幂等，所以：

- 服务端哪天不再发 BOM，补丁自动变成空操作，无需卸载；
- 卸载插件后**不还原**补丁——还原反而会影响同样在用 jmcomic 的其它插件和在途请求。

**两个实现细节（都踩过坑）**：

1. HTTP 层是 `curl_cffi` 而不是 `requests`：它的 `content` 是**普通实例属性**，
   直接 `resp.content = ...` 即可；老版本 `requests` 的 `content` 是只读
   property，所以要回退去写 `_content`。两者都处理了。
2. `text` 是**带缓存的** property（`curl_cffi` 缓存到 `self._text`）。
   只改 `content` 不失效 `_text`，校验读到的仍是带 BOM 的旧值，
   **补丁等于没打**——这是第一版失败的原因，单元测试有专门的用例守着。

### 能力声明

MaiBot 的能力授权是**精确字符串匹配**（`AuthorizationManager.check_capability`），
没有前缀或伞形展开，**不存在 `send_message` 这种总括能力**。
因此 `_manifest.json` 必须逐条声明实际用到的发送方法：

```json
"capabilities": ["send.text", "send.image", "send.custom", "send.hybrid"]
```

少声明任何一个，运行时都会被 `E_CAPABILITY_DENIED` 拒绝。
单元测试 `test_capabilities_cover_used_send_methods`
会自动比对源码用到的 `ctx.send.*` 与清单声明，防止漏声明。

### PDF 为什么传本地路径而不是 base64

文件段整条链上，**只有 `url` 这一个键名是 Host 与适配器都认的**：

| 环节 | 读取 / 输出的键 |
|------|----------------|
| Host `FileComponent.from_payload` / `to_payload` | `base64`、`url` |
| NapCat 适配器 `_build_file_segment` | `file`、`path`、`url` |

两个错误写法及实际后果：

- `send.custom("file", {"base64": ...})` —— Host 会正常序列化，
  但适配器**不读 `base64`**，取不到文件引用，最终降级成字面文本 `[file]`。
- `send.custom("dict", {"type": "file", "data": {"file": <本地路径>}})` ——
  Host 的 `PluginMessageUtils._component_from_dict` 看到内层 `type == "file"`
  就把它转成 `FileComponent`，而 `from_payload` 只认 `base64` / `url`，
  本地路径键被丢掉，同样降级成 `[file]`。

所以文件引用必须放在 `url` 里，而**值用本地绝对路径**（本插件采用）：

```python
await self.ctx.send.custom(
    "file",
    {"name": pdf_name, "url": pdf_path.resolve().as_posix()},
    stream_id,
)
```

**为什么不用 `base64://`**：base64 得整个塞进一次 RPC 调用，而 Runner 的帧上限是
`MAX_FRAME_SIZE`（16 MB）。40 页 PDF 的 base64 约 29 MB，会直接抛
`E_UNKNOWN 帧大小 29610142 超过最大限制 16777216`。本地路径只是一个短字符串，
与文件大小无关。

适配器的 `_normalize_file_reference` 会给没有前缀的值补上 `file://`，
NapCat 的 `checkUriType` 再用 `fs.existsSync` 判定为 Local 并**直接使用该路径**
（不下载、不走下载代理），因此：

- 插件与 NapCat 必须同机；
- 文件在发送期间必须存在（见下面的清理延迟）。

### PDF 体积控制

目标：**60 页的 PDF 约 10 MB**，即单页 JPEG 预算
`10 MB / 60 ≈ 171 KB`（`pdf_target_mb_per_60_pages` 可调，按页均摊）。

压缩顺序**优先保画质**：

1. 长边缩到 `pdf_max_resolution`，按 `pdf_jpeg_quality` 编码；
   **不超预算就直接用**，此时画质最好、完全不降质。
2. 超预算则**保持质量、按面积比例降分辨率**（JPEG 体积≈与像素数成正比），
   最多迭代 4 次。
3. 分辨率已到 `PDF_MIN_RESOLUTION`（900px）仍超预算，才逐步降质量到
   `PDF_MIN_JPEG_QUALITY`（40）。

即"宁可略微降分辨率，也不把质量压到出块"。实测（1500×2100 合成页，12 张互不相同，
用 MaiBot 的 python-env 跑真实 reportlab + Pillow）：

| 页面类型 | 单页 @q75 | 旧参数（无预算） | 新参数（10MB/60页） |
|---------|----------|----------------|-------------------|
| 线条+轻颗粒 | 706 KB | 347 KB/页 → 60页 ≈ 20.3 MB | **160 KB/页 → 60页 ≈ 9.4 MB**（1050px，q75） |
| 线条+重颗粒 | 1060 KB | 450 KB/页 → 60页 ≈ 26.4 MB | **142 KB/页 → 60页 ≈ 8.3 MB**（925px，q75） |

每次生成都会打一行统计，便于核对实际效果：

```
PDF 压缩统计：60 页，单页 152~160 KB（平均 158 KB），长边 1050~1055 px，质量 75~75
```

**必须关掉 reportlab 的 ASCII85**：默认它给内嵌 JPEG 外面再套一层 ASCII85
（过滤器变成 `['ASCII85Decode', 'DCTDecode']`），凭空多出约 25% 体积。
`create_pdf_from_images` 里设置 `rl_config.useA85 = 0` 后过滤器只剩 `DCTDecode`，
JPEG 原样嵌入，实测 735,913 字节的 JPEG：**1.252× → 1.002×**。
该开关是 reportlab 的进程级配置，但只改变编码方式、不影响 PDF 有效性。

另外**不要用渐进式 JPEG**（`progressive=True`）：reportlab 认不出它，
会退化成解码后重编码，体积和画质都失控。

### 后台任务与超时

组件 RPC 的默认超时是 60 秒（`DEFAULT_COMPONENT_RPC_TIMEOUT_MS`），
而下载整章再合成 PDF 通常更久。所以 `/jm` 命令只负责受理并立即返回，
真正的下载与发送放在后台任务里完成，卸载时会被取消。

### PDF 清理延迟

文件消息传给平台的是本地路径，NapCat 判定为 Local 后直接读取该文件，
发完立刻删除可能让上传读不到内容。已发送的 PDF 会保留
`_PDF_KEEP_SECONDS`（300 秒）后再删除，插件卸载时也会清理残留。

## 测试方式

### 静态检查（无需启动 MaiBot）

```bash
# 1. 校验 _manifest.json 是合法 JSON
python -c "import json; json.load(open('_manifest.json', encoding='utf-8')); print('manifest OK')"

# 2. 校验 plugin.py 语法（不执行导入）
python -m py_compile plugin.py
```

`tests/` 目录不随插件发布，仅在开发副本中保留（4 个文件：`test_manifest.py`、
`test_batching.py`、`test_queue_serial.py`、`test_jmcomic_live.py`）。
前两个只依赖标准库，后两个需要 `maibot_sdk`，用 MaiBot 自带的解释器运行：

```bash
<MaiBot>/python-env/python.exe tests/test_queue_serial.py
<MaiBot>/python-env/python.exe tests/test_jmcomic_live.py
```

### 本地加载

把插件放入 `plugins/`，启动 MaiBot，观察日志中出现 `JMComic 漫画插件已加载`，且插件管理页能看到该插件。

### 功能测试

1. 在私聊发送 `/jm help`，确认返回帮助文本。
2. 发送 `/jm <漫画ID>`，确认下载并发回 PDF。
3. 发送 `/jm <漫画ID> 1 jpg`，确认按图片发送。
4. **排队验证**：发送 `/jm <漫画ID>` 后立刻再发一条 `/jm <另一个ID>`，
   确认第二条回复 `已加入待处理队列（前面还有 1 个）...`，
   且它的 `正在下载《...》` 出现在第一条的 PDF **发送完成之后**。
5. 连续发 5 条命令，确认第 4 条起回复 `待处理队列已满`。
6. 让 LLM 回答「帮我搜一下某部漫画」，确认能触发 `jmcomic_search_album` 工具并返回结果。
7. 禁用或卸载插件，确认下载临时目录被清理、无后台任务残留。

## 变更说明（相对旧版 v1.x）

- 从旧 `src.plugin_system`（`@register_plugin` / `BaseCommand`）迁移到 `maibot-plugin-sdk`（`MaiBotPlugin` + `@Command` / `@Tool`）。
- `_manifest.json` 升级到 `manifest_version: 2`，依赖通过 `dependencies` 声明，移除运行时 `pip install` 逻辑。
- 移除旧版对主程序 `MessageSending` 的 monkey-patch（跳过 VLM），改用 SDK 官方发送能力。
- 修复旧版 `search_album` 未接线的问题，搜索改为可通过 `@Tool` 正常调用。
- 移除未使用的加密 ZIP 死代码；下载改为基于 `JmOption`（不再 `os.chdir`，避免线程间工作目录污染）。
- 补齐代理、自定义域名、账号登录的实际生效逻辑（旧版仅声明未生效）。

### 2.0.0 修订（PDF 发送与命令调度）

- **修正能力声明**：原先声明的 `send_message` 并非有效能力名，会导致所有发送被
  `E_CAPABILITY_DENIED` 拒绝；现改为逐条声明 `send.text` / `send.image` /
  `send.custom` / `send.hybrid`，并加了防漏声明的回归测试。
- **修正 PDF 发送通道**：PDF 数据必须放在 `url` 字段（`url="base64://<数据>"`），
  因为只有 `url` 是 Host 与 NapCat 适配器都认的键名；
  放 `base64` 会被适配器丢弃，放 `file` 会被 Host 丢成空文件段
  （两者都会降级成字面文本 `[file]`）。详见上文说明。
- **取消静默退化为图片**：PDF 发送失败时不再自动改发图片，而是明确提示；
  如需旧行为可开启 `features.pdf_fallback_to_images`。
- **规避 60 秒组件超时**：命令改为受理即返回，下载与发送在后台任务中完成，
  不再出现 `E_TIMEOUT ... invoke_command 超时 (60000ms)`。
- **PDF 延后清理**：发送后保留 `_PDF_KEEP_SECONDS`（现为 300 秒）再删除，
  避免平台读取本地文件时已被清理。该清理**已异步化**，不再阻塞后续命令。
- **PDF 分卷规则**：页数 ≤ `single_pdf_max_pages`（默认 60）时合成单个 PDF；
  超过则按 `images_per_pdf`（默认 40）拆成多个分别发送。
- **PDF 体积控制**：按 `pdf_target_mb_per_60_pages`（默认 10 MB / 60 页）均摊出单页预算，
  优先保质量、其次降分辨率、最后才降质量；并关掉 reportlab 的 ASCII85 编码省下约 25%。
- **PDF 改用本地路径发送**：不再走 `base64://`（会撞 16 MB 帧上限）。
- **提示文本**：受理后发 `正在下载《作品名》第X章`，生成 PDF 前发
  `共 x 张图片，将分为 y 个 PDF 发送`（单卷时只说 `共 x 页`）。
  两段文本分别在纯函数 `_format_download_notice` / `_format_page_notice` 里。
- **命令串行化**：新增待处理队列，前一个命令的 PDF 完全发完之前，
  后续命令只排队不并发（`max_pending_jobs` 默认 3）；
  同时把 300 秒的 PDF 延迟清理移出任务关键路径，避免队列被拖住。
- **禁漫 UTF-8 BOM 兼容**：服务端在响应体前加了 BOM 导致 jmcomic 全部判成
  「不是 json」并重试到失败（2.7.7 为最新版，上游未修）。插件侧打补丁剥离前导 BOM。
- **失败提示不再刷屏**：jmcomic 会把 24 次重试详情塞进异常文本，原先整段发到聊天里；
  现只发首行并截断（180 字），完整堆栈仍进日志。
- **命令串行化**：新增待处理队列，前一个命令的 PDF 完全发完之前，
  后续命令只排队不并发（`max_pending_jobs` 默认 3）；
  同时把 300 秒的 PDF 延迟清理移出任务关键路径，避免队列被拖住。

## 依赖项目

- [JMComic-Crawler-Python](https://github.com/hect0x7/JMComic-Crawler-Python)
- ReportLab
- Pillow
