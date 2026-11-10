"""优云智算视频生成(MiniMax H3)与超分(Flash VSR)命令行工具.

密钥使用环境变量 UCLOUD_API_KEY.

reference:
    https://www.compshare.cn/docs/minimax-h3/api/minimax-h3-video-api
    https://www.compshare.cn/docs/modelverse/models/video_api/flash-vsr-video-upscaling-api
"""

import base64
import logging
import mimetypes
import os
import signal
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple, cast

import requests
import typer

__all__ = [
    "app",
]


logger = logging.getLogger(__name__)


# API 请求超时, 单位为秒
_REQUEST_TIMEOUT = 60
# 视频下载超时, 单位为秒
_DOWNLOAD_TIMEOUT = 600
# HTTP 状态码达到该值时判定为错误
_HTTP_ERROR_THRESHOLD = 400
# 任务轮询间隔, 单位为秒
_POLL_INTERVAL = 5
# 等待任务完成的默认超时, 单位为秒
_DEFAULT_TIMEOUT = 1800
# 终态任务状态集合
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

# 媒体文件后缀名 -> MIME 类型, 未命中时回退到 mimetypes 猜测
_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}
_AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}

# 各类素材单文件大小上限, 单位为字节
_MAX_IMAGE_SIZE = 30 * 1024 * 1024
_MAX_VIDEO_SIZE = 50 * 1024 * 1024
_MAX_AUDIO_SIZE = 15 * 1024 * 1024

# 默认参数值
_DEFAULT_BASE_URL = "https://cp.compshare.cn"
_DEFAULT_MODEL = "MiniMax-H3"
_DEFAULT_RESOLUTION = "768P"
_DEFAULT_DURATION = 5
_DEFAULT_RATIO = "16:9"
_DEFAULT_PROMPT_TYPE = "text"


class _Endpoints(NamedTuple):
    """异步任务类接口的创建/查询/取消路径, 查询与取消路径含 {task_id} 占位符."""

    create: str
    query: str
    cancel: str


# MiniMax H3 视频生成接口
_GENERATION_ENDPOINTS = _Endpoints(
    create="/minimax/v2/video_generation",
    query="/minimax/v2/query/video_generation/{task_id}",
    cancel="/minimax/v2/video_generation/{task_id}",
)
# Flash VSR 视频超分接口
_UPSCALING_ENDPOINTS = _Endpoints(
    create="/flashvsr/v1/video_upscaling",
    query="/flashvsr/v1/query/video_upscaling/{task_id}",
    cancel="/flashvsr/v1/video_upscaling/{task_id}",
)

# Flash VSR 参数
_UPSCALING_MODEL = "FlashVSR"
_DEFAULT_UPSCALING_RESOLUTION = "2K"
_UPSCALING_VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}
_MAX_UPSCALING_VIDEO_SIZE = 50 * 1024 * 1024


# ======================================================================================
# Fields
# ======================================================================================
PromptTypeField = Annotated[
    Literal["text", "file"],
    typer.Option(help="prompt 类型: text 为直接传入的文本, file 为 prompt 文件路径."),
]
PromptField = Annotated[
    str,
    typer.Option(
        help="prompt 内容: prompt_type 为 text 时是提示词文本, 为 file 时是文件路径."
    ),
]
I2vPromptField = Annotated[
    str,
    typer.Option(
        help="提示词, 可选; 首尾帧模式下允许省略文本.",
    ),
]
FirstFrameField = Annotated[
    str,
    typer.Option(
        help="首帧图片, 支持公网 URL 或本地文件路径(本地文件自动转为 Data URL)."
    ),
]
LastFrameField = Annotated[
    str | None,
    typer.Option(help="尾帧图片, 支持公网 URL 或本地文件路径."),
]
ImageField = Annotated[
    list[str],
    typer.Option(
        "--image",
        help="参考图片, 支持公网 URL 或本地文件路径, 可多次指定, 最多 9 张.",
    ),
]
VideoField = Annotated[
    list[str],
    typer.Option(
        "--video",
        help="参考视频, 支持公网 URL 或本地文件路径, 可多次指定, 最多 3 个.",
    ),
]
AudioField = Annotated[
    list[str],
    typer.Option(
        "--audio",
        help="参考音频, 支持公网 URL 或本地文件路径, 可多次指定, 最多 3 个.",
    ),
]
OutputField = Annotated[
    str | None,
    typer.Option(help="视频保存路径, 未指定时自动创建临时文件."),
]
TimeoutField = Annotated[
    int,
    typer.Option(
        help="等待任务完成的超时时间(秒), 超时后自动取消任务.",
        min=1,
    ),
]
ModelField = Annotated[
    Literal["MiniMax-H3", "minimax-h3-lite"],
    typer.Option(help="模型名称."),
]
ResolutionField = Annotated[
    Literal["480P", "768P", "1080P", "2K", "4K"],
    typer.Option(help="输出分辨率档位, 区分大小写; 1080P/2K/4K 启用后置超分."),
]
DurationField = Annotated[
    int,
    typer.Option(help="视频时长(秒).", min=4, max=30),
]
RatioField = Annotated[
    Literal["16:9", "21:9", "4:3", "1:1", "3:4", "9:16", "adaptive"],
    typer.Option(help="宽高比; 有图片素材时可传 adaptive 自动匹配素材比例."),
]
UseContextIrField = Annotated[
    bool,
    typer.Option(
        "--use-context-ir/--no-use-context-ir",
        help="是否启用 Context-IR 提示词优化.",
    ),
]
MuteAudioField = Annotated[
    bool,
    typer.Option("--mute-audio/--no-mute-audio", help="是否移除生成视频的音轨."),
]
AigcWatermarkField = Annotated[
    bool,
    typer.Option(
        "--aigc-watermark/--no-aigc-watermark",
        help="是否添加 AIGC 水印.",
    ),
]
BaseUrlField = Annotated[
    str,
    typer.Option(help="请求端点前缀."),
]
UpscaleVideoField = Annotated[
    str,
    typer.Option(
        help="待超分的原视频, 支持公网 URL 或本地文件路径(本地文件自动转为 Data URL); "
        "支持 MP4/MOV/M4V/MKV/AVI, 不超过 50 MiB."
    ),
]
UpscaleResolutionField = Annotated[
    Literal["720p", "1080p", "2K", "4K"],
    typer.Option(help="目标分辨率档位, 需对原视频产生实际放大效果."),
]


# ======================================================================================
# Helpers
# ======================================================================================
def _validate_api_key() -> str:
    """从环境变量读取 API Key, 未设置时抛出 BadParameter."""
    api_key = os.environ.get("UCLOUD_API_KEY")
    if api_key is None:
        errmsg = "未找到 API Key, 请设置环境变量 UCLOUD_API_KEY"
        raise typer.BadParameter(errmsg)
    return api_key


def _resolve_prompt(
    *,
    prompt: str,
    prompt_type: str,
) -> str:
    """根据 prompt_type 解析 prompt, file 类型时从文件中读取内容."""
    if prompt_type == "text":
        return prompt
    try:
        return Path(prompt).read_text(encoding="utf-8")
    except OSError as e:
        errmsg = f"无法读取 prompt 文件: {prompt!r} ({e})"
        raise typer.BadParameter(errmsg) from None


def _resolve_prompt_or_empty(
    *,
    prompt: str,
    prompt_type: str,
) -> str:
    """解析 prompt, 值为空字符串时直接返回空, 不读取文件."""
    if not prompt:
        return ""
    return _resolve_prompt(prompt=prompt, prompt_type=prompt_type)


def _guess_mime(
    *,
    path: Path,
    mime_types: dict[str, str],
) -> str:
    """根据文件后缀名猜测 MIME 类型, 优先查表, 未命中时回退到 mimetypes."""
    mime = mime_types.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    if mime is None:
        errmsg = f"无法识别的媒体文件类型: {path!r}"
        raise typer.BadParameter(errmsg)
    return mime


def _resolve_media_url(
    *,
    value: str,
    mime_types: dict[str, str],
    max_size: int,
) -> str:
    """解析素材地址: 公网 URL 原样返回, 本地文件读取后转为 Data URL."""
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    try:
        content = path.read_bytes()
    except OSError as e:
        errmsg = f"无法读取媒体文件: {value!r} ({e})"
        raise typer.BadParameter(errmsg) from None
    if len(content) > max_size:
        errmsg = (
            f"媒体文件超过大小上限 {max_size} 字节: {value!r} ({len(content)} 字节)"
        )
        raise typer.BadParameter(errmsg)
    mime = _guess_mime(path=path, mime_types=mime_types)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _resolve_media_urls(
    *,
    values: list[str],
    mime_types: dict[str, str],
    max_size: int,
    limit: int,
) -> list[str]:
    """批量解析素材地址, 超出数量上限时抛出 BadParameter."""
    if len(values) > limit:
        errmsg = f"素材数量超过上限 {limit}: {len(values)}"
        raise typer.BadParameter(errmsg)
    return [
        _resolve_media_url(value=value, mime_types=mime_types, max_size=max_size)
        for value in values
    ]


def _api_request(
    *,
    method: str,
    url: str,
    api_key: str,
    json_payload: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """发起带鉴权的 JSON API 请求, 返回响应 JSON, 非 2xx 时抛出 RuntimeError."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        **(extra_headers or {}),
    }
    resp = requests.request(
        method,
        url,
        headers=headers,
        json=json_payload,
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code >= _HTTP_ERROR_THRESHOLD:
        errmsg = (
            f"API request failed: method={method!r}, url={url!r}, "
            f"status={resp.status_code!r}, body={resp.text!r}"
        )
        raise RuntimeError(errmsg)
    try:
        data: dict[str, Any] = resp.json()
    except ValueError as e:
        errmsg = f"API response is not valid JSON: url={url!r}, body={resp.text!r}"
        raise RuntimeError(errmsg) from e
    # NOTE: 平台业务错误以 HTTP 200 + RetCode/Message 响应体返回
    if data.get("RetCode"):
        errmsg = (
            f"API request failed: method={method!r}, url={url!r}, "
            f"retcode={data.get('RetCode')!r}, message={data.get('Message')!r}"
        )
        raise RuntimeError(errmsg)
    return data


def _create_task(
    *,
    base_url: str,
    api_key: str,
    endpoints: _Endpoints,
    payload: dict[str, Any],
) -> str:
    """创建异步任务, 返回 task_id."""
    logger.info(
        "Creating task: model=%r, resolution=%r, duration=%r",
        payload.get("model"),
        payload.get("resolution"),
        payload.get("duration"),
    )
    resp = _api_request(
        method="POST",
        url=f"{base_url}{endpoints.create}",
        api_key=api_key,
        json_payload=payload,
        extra_headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    task_id = resp.get("task_id")
    if not task_id:
        errmsg = f"No task_id in creation response: {resp!r}"
        raise RuntimeError(errmsg)
    return str(task_id)


def _query_task(
    *,
    base_url: str,
    api_key: str,
    endpoints: _Endpoints,
    task_id: str,
) -> dict[str, Any]:
    """查询单个任务, 返回 task 对象."""
    resp = _api_request(
        method="GET",
        url=f"{base_url}{endpoints.query.format(task_id=task_id)}",
        api_key=api_key,
    )
    task: Any = resp.get("task")
    if not isinstance(task, dict):
        errmsg = f"No task object in query response: {resp!r}"
        raise TypeError(errmsg)
    return cast("dict[str, Any]", task)


def _cancel_task(
    *,
    base_url: str,
    api_key: str,
    endpoints: _Endpoints,
    task_id: str,
) -> None:
    """取消指定任务."""
    logger.warning("Cancelling task: task_id=%r", task_id)
    _api_request(
        method="DELETE",
        url=f"{base_url}{endpoints.cancel.format(task_id=task_id)}",
        api_key=api_key,
    )


def _describe_error(
    *,
    task: dict[str, Any],
) -> str:
    """从任务对象中提取失败信息描述."""
    error: dict[str, Any] = task.get("error") or {}
    return f"code={error.get('code')!r}, message={error.get('message')!r}"


def _wait_task(
    *,
    base_url: str,
    api_key: str,
    endpoints: _Endpoints,
    task_id: str,
    timeout: int,
) -> dict[str, Any]:
    """轮询任务直到终态, 返回最终 task 对象.

    收到 SIGINT/SIGTERM 或等待超时时自动取消任务.
    """
    cancelled = False

    def _cancel_quietly() -> None:
        nonlocal cancelled
        if cancelled:
            return
        cancelled = True
        try:
            _cancel_task(
                base_url=base_url,
                api_key=api_key,
                endpoints=endpoints,
                task_id=task_id,
            )
        except Exception:
            logger.exception("Failed to cancel task: task_id=%r", task_id)

    def _on_signal(signum: int, _frame: object) -> None:
        logger.warning(
            "Received exit signal, cancelling task: signal=%r, task_id=%r",
            signum,
            task_id,
        )
        _cancel_quietly()
        typer.echo(f"\n收到退出信号, 任务已取消: task_id={task_id}")
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        deadline = time.monotonic() + timeout
        while True:
            task = _query_task(
                base_url=base_url,
                api_key=api_key,
                endpoints=endpoints,
                task_id=task_id,
            )
            status = task.get("status")
            if status == "succeeded":
                return task
            if status in _TERMINAL_STATUSES:
                errmsg = (
                    f"Task terminated: task_id={task_id!r}, "
                    f"status={status!r}, {_describe_error(task=task)}"
                )
                raise RuntimeError(errmsg)
            if time.monotonic() >= deadline:
                _cancel_quietly()
                errmsg = (
                    f"Task timed out and has been cancelled: "
                    f"task_id={task_id!r}, timeout={timeout!r}"
                )
                raise RuntimeError(errmsg)
            logger.info(
                "Waiting for task: task_id=%r, status=%r, "
                "estimated_remaining_seconds=%r",
                task_id,
                status,
                task.get("estimated_remaining_seconds"),
            )
            time.sleep(_POLL_INTERVAL)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if cancelled:
            typer.echo(f"任务已取消: task_id={task_id}")


def _download_video(
    *,
    url: str,
) -> bytes:
    """下载生成完成的视频, 返回二进制内容."""
    resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
    if resp.status_code >= _HTTP_ERROR_THRESHOLD:
        errmsg = f"Failed to download video: url={url!r}, status={resp.status_code!r}"
        raise RuntimeError(errmsg)
    return resp.content


def _save_video(
    *,
    video: bytes,
    output: str | None,
    prefix: str,
) -> Path:
    """保存视频, 未指定输出路径时创建临时文件."""
    if output is None:
        with tempfile.NamedTemporaryFile(
            prefix=prefix,
            suffix=".mp4",
            delete=False,
        ) as temp_file:
            path = Path(temp_file.name)
        logger.info("Created temporary output file: path=%r", path)
    else:
        path = Path(output)

    path.write_bytes(video)
    typer.echo(f"视频已保存: {path} ({len(video)} bytes)")
    return path


def _submit_and_wait(  # noqa: PLR0913
    *,
    base_url: str,
    api_key: str,
    endpoints: _Endpoints,
    payload: dict[str, Any],
    output: str | None,
    timeout: int,
    prefix: str,
) -> None:
    """提交任务并挂起等待, 成功后下载视频并保存到本地."""
    task_id = _create_task(
        base_url=base_url,
        api_key=api_key,
        endpoints=endpoints,
        payload=payload,
    )
    typer.echo(f"任务已提交: task_id={task_id}")
    try:
        task = _wait_task(
            base_url=base_url,
            api_key=api_key,
            endpoints=endpoints,
            task_id=task_id,
            timeout=timeout,
        )
    except Exception:
        logger.exception("Task failed: task_id=%r", task_id)
        raise typer.Exit(code=1) from None
    content: Any = task.get("content")
    url: str | None = content.get("url") if content else None
    if not url:
        logger.error(
            "No video url in succeeded task: task_id=%r, task=%r", task_id, task
        )
        raise typer.Exit(code=1) from None
    video = _download_video(url=url)
    _save_video(video=video, output=output, prefix=prefix)


def _build_text_content(
    *,
    prompt: str,
) -> dict[str, Any]:
    """构建 text 内容元素."""
    return {"type": "text", "text": prompt}


def _build_image_content(
    *,
    url: str,
    role: str,
) -> dict[str, Any]:
    """构建 image_url 内容元素."""
    return {"type": "image_url", "image_url": {"url": url}, "role": role}


def _build_media_content(
    *,
    url: str,
    media_type: str,
    role: str,
) -> dict[str, Any]:
    """构建 video_url/audio_url 内容元素."""
    return {
        "type": media_type,
        media_type: {"url": url},
        "role": role,
    }


def _submit(  # noqa: PLR0913
    *,
    payload: dict[str, Any],
    output: str | None,
    timeout: int,
    base_url: str,
    prefix: str,
    endpoints: _Endpoints = _GENERATION_ENDPOINTS,
) -> None:
    """校验密钥后提交任务并等待完成."""
    api_key = _validate_api_key()
    _submit_and_wait(
        base_url=base_url,
        api_key=api_key,
        endpoints=endpoints,
        payload=payload,
        output=output,
        timeout=timeout,
        prefix=prefix,
    )


# ======================================================================================
# Commands
# ======================================================================================
app = typer.Typer(
    name="ucloud",
    help="视频生成/超分(优云智算 MiniMax H3 / Flash VSR).",
    no_args_is_help=True,
)


@app.command()
def t2v(  # noqa: PLR0913
    *,
    prompt: PromptField,
    prompt_type: PromptTypeField = _DEFAULT_PROMPT_TYPE,
    output: OutputField = None,
    timeout: TimeoutField = _DEFAULT_TIMEOUT,
    model: ModelField = _DEFAULT_MODEL,
    resolution: ResolutionField = _DEFAULT_RESOLUTION,
    duration: DurationField = _DEFAULT_DURATION,
    ratio: RatioField = _DEFAULT_RATIO,
    use_context_ir: UseContextIrField = False,
    mute_audio: MuteAudioField = False,
    aigc_watermark: AigcWatermarkField = False,
    base_url: BaseUrlField = _DEFAULT_BASE_URL,
) -> None:
    """文生视频: 通过提示词生成视频."""
    text = _resolve_prompt(prompt=prompt, prompt_type=prompt_type)
    if not text.strip():
        errmsg = "文生视频必须包含非空提示词"
        raise typer.BadParameter(errmsg)

    payload: dict[str, Any] = {
        "model": model,
        "content": [_build_text_content(prompt=text)],
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
        "use_context_ir": use_context_ir,
        "mute_audio": mute_audio,
        "aigc_watermark": aigc_watermark,
    }
    _submit(
        payload=payload,
        output=output,
        timeout=timeout,
        base_url=base_url,
        prefix="pymzckcli_t2v_",
    )


@app.command()
def i2v(  # noqa: PLR0913
    *,
    first_frame: FirstFrameField,
    last_frame: LastFrameField = None,
    prompt: I2vPromptField = "",
    prompt_type: PromptTypeField = _DEFAULT_PROMPT_TYPE,
    output: OutputField = None,
    timeout: TimeoutField = _DEFAULT_TIMEOUT,
    model: ModelField = _DEFAULT_MODEL,
    resolution: ResolutionField = _DEFAULT_RESOLUTION,
    duration: DurationField = _DEFAULT_DURATION,
    ratio: RatioField = _DEFAULT_RATIO,
    use_context_ir: UseContextIrField = False,
    mute_audio: MuteAudioField = False,
    aigc_watermark: AigcWatermarkField = False,
    base_url: BaseUrlField = _DEFAULT_BASE_URL,
) -> None:
    """首尾帧生视频: 基于首帧(可选尾帧)图片生成视频."""
    content: list[dict[str, Any]] = []
    text = _resolve_prompt_or_empty(prompt=prompt, prompt_type=prompt_type)
    if text:
        content.append(_build_text_content(prompt=text))
    content.append(
        _build_image_content(
            url=_resolve_media_url(
                value=first_frame,
                mime_types=_IMAGE_MIME_TYPES,
                max_size=_MAX_IMAGE_SIZE,
            ),
            role="first_frame",
        )
    )
    if last_frame is not None:
        content.append(
            _build_image_content(
                url=_resolve_media_url(
                    value=last_frame,
                    mime_types=_IMAGE_MIME_TYPES,
                    max_size=_MAX_IMAGE_SIZE,
                ),
                role="last_frame",
            )
        )

    payload: dict[str, Any] = {
        "model": model,
        "content": content,
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
        "use_context_ir": use_context_ir,
        "mute_audio": mute_audio,
        "aigc_watermark": aigc_watermark,
    }
    _submit(
        payload=payload,
        output=output,
        timeout=timeout,
        base_url=base_url,
        prefix="pymzckcli_i2v_",
    )


@app.command()
def r2v(  # noqa: PLR0913
    *,
    prompt: PromptField,
    prompt_type: PromptTypeField = _DEFAULT_PROMPT_TYPE,
    image: ImageField = [],  # noqa: B006
    video: VideoField = [],  # noqa: B006
    audio: AudioField = [],  # noqa: B006
    output: OutputField = None,
    timeout: TimeoutField = _DEFAULT_TIMEOUT,
    model: ModelField = _DEFAULT_MODEL,
    resolution: ResolutionField = _DEFAULT_RESOLUTION,
    duration: DurationField = _DEFAULT_DURATION,
    ratio: RatioField = _DEFAULT_RATIO,
    use_context_ir: UseContextIrField = False,
    mute_audio: MuteAudioField = False,
    aigc_watermark: AigcWatermarkField = False,
    base_url: BaseUrlField = _DEFAULT_BASE_URL,
) -> None:
    """全能参考生视频: 基于参考图片/视频/音频组合生成视频.

    提示词中可用 <Picture N>/<Video N>/<Audio N> 标签引用对应素材,
    三类素材各自从 1 开始编号, 按传入顺序对应.
    """
    if not image and not video:
        if audio:
            errmsg = "参考音频不能单独使用, 至少需要一张参考图片或一个参考视频"
            raise typer.BadParameter(errmsg)
        errmsg = "全能参考模式至少需要一项参考素材(--image/--video/--audio)"
        raise typer.BadParameter(errmsg)

    text = _resolve_prompt(prompt=prompt, prompt_type=prompt_type)
    content: list[dict[str, Any]] = [_build_text_content(prompt=text)]
    content.extend(
        _build_image_content(url=url, role="reference_image")
        for url in _resolve_media_urls(
            values=image,
            mime_types=_IMAGE_MIME_TYPES,
            max_size=_MAX_IMAGE_SIZE,
            limit=9,
        )
    )
    content.extend(
        _build_media_content(url=url, media_type="video_url", role="reference_video")
        for url in _resolve_media_urls(
            values=video,
            mime_types=_VIDEO_MIME_TYPES,
            max_size=_MAX_VIDEO_SIZE,
            limit=3,
        )
    )
    content.extend(
        _build_media_content(url=url, media_type="audio_url", role="reference_audio")
        for url in _resolve_media_urls(
            values=audio,
            mime_types=_AUDIO_MIME_TYPES,
            max_size=_MAX_AUDIO_SIZE,
            limit=3,
        )
    )

    payload: dict[str, Any] = {
        "model": model,
        "content": content,
        "resolution": resolution,
        "duration": duration,
        "ratio": ratio,
        "use_context_ir": use_context_ir,
        "mute_audio": mute_audio,
        "aigc_watermark": aigc_watermark,
    }
    _submit(
        payload=payload,
        output=output,
        timeout=timeout,
        base_url=base_url,
        prefix="pymzckcli_r2v_",
    )


@app.command()
def upscale(
    *,
    video: UpscaleVideoField,
    resolution: UpscaleResolutionField = _DEFAULT_UPSCALING_RESOLUTION,
    output: OutputField = None,
    timeout: TimeoutField = _DEFAULT_TIMEOUT,
    base_url: BaseUrlField = _DEFAULT_BASE_URL,
) -> None:
    """视频超分(Flash VSR): 将已有视频提升到目标分辨率档位."""
    payload: dict[str, Any] = {
        "model": _UPSCALING_MODEL,
        "video_url": _resolve_media_url(
            value=video,
            mime_types=_UPSCALING_VIDEO_MIME_TYPES,
            max_size=_MAX_UPSCALING_VIDEO_SIZE,
        ),
        "resolution": resolution,
    }
    _submit(
        payload=payload,
        output=output,
        timeout=timeout,
        base_url=base_url,
        prefix="pymzckcli_upscale_",
        endpoints=_UPSCALING_ENDPOINTS,
    )
