"""OpenAI Images API 兼容的图片生成/编辑工具.

密钥使用环境变量 OPENAI_API_KEY.
"""

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
import typer

__all__ = [
    "app",
]


logger = logging.getLogger(__name__)


# 图片生成耗时较长, 请求超时放宽到 10 分钟
_REQUEST_TIMEOUT = 600

# HTTP 状态码错误判定阈值
_HTTP_ERROR_THRESHOLD = 400

# 图片文件后缀名 -> MIME 类型
_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

# 输出格式 -> 临时文件后缀名
_FORMAT_SUFFIXES = {
    "png": ".png",
    "jpeg": ".jpg",
    "webp": ".webp",
}

# 默认参数值
_DEFAULT_PROMPT_TYPE = "text"
_DEFAULT_MODEL = "gpt-image-2.5-flare"
_DEFAULT_SIZE = "1024x1024"
_DEFAULT_QUALITY = "low"
_DEFAULT_OUTPUT_FORMAT = "png"
_DEFAULT_BACKGROUND = "auto"
_DEFAULT_MODERATION = "auto"
_DEFAULT_BASE_URL = "https://api.teamorouter.com/v1"

# 图片生成与编辑相关路径
_GENERATIONS_PATH = "/images/generations"
_EDITS_PATH = "/images/edits"


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
ImageField = Annotated[
    list[str],
    typer.Option(
        "--image",
        help="原图文件路径(png/jpeg/webp), 可多次指定以传入多张参考图.",
    ),
]
MaskField = Annotated[
    str | None,
    typer.Option(help="遮罩图文件路径, 透明区域为可编辑区域, 尺寸需与原图一致."),
]
OutputField = Annotated[
    str | None,
    typer.Option(help="图片保存路径, 未指定时自动创建临时文件."),
]
ModelField = Annotated[
    Literal["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"],
    typer.Option(help="模型名称."),
]
# 常用尺寸白名单, 均满足 API 约束: 宽高为 16 的倍数, 长宽比不超过 3:1,
# 总像素数在 655,360 到 8,294,400 之间, 最大边不超过 3840 像素
SizeField = Annotated[
    Literal[
        "auto",
        "1024x1024",
        "1536x1024",
        "1024x1536",
        "2048x2048",
        "2048x1152",
        "1152x2048",
        "3840x2160",
        "2160x3840",
    ],
    typer.Option(help="图片尺寸."),
]
QualityField = Annotated[
    Literal["auto", "low", "medium", "high"],
    typer.Option(help="生成质量."),
]
InputFidelityField = Annotated[
    Literal["high", "low"] | None,
    typer.Option(help="对原图细节的保真程度, 未指定时由服务端决定."),
]
OutputFormatField = Annotated[
    Literal["png", "jpeg", "webp"],
    typer.Option(help="输出图片格式."),
]
EditOutputFormatField = Annotated[
    Literal["png", "jpeg", "webp"],
    typer.Option(help="输出图片格式; NOTE: 编辑接口实测返回的图片不带透明通道."),
]
OutputCompressionField = Annotated[
    int | None,
    typer.Option(
        help="JPEG/WebP 压缩等级, 取值范围 [0, 100], 仅对 jpeg/webp 生效.",
        min=0,
        max=100,
    ),
]
BackgroundField = Annotated[
    Literal["auto", "transparent", "opaque"],
    typer.Option(help="背景; transparent 需搭配 png/webp 格式."),
]
EditBackgroundField = Annotated[
    Literal["auto", "transparent", "opaque"],
    typer.Option(help="背景; 编辑接口实测返回的图片不带透明通道."),
]
ModerationField = Annotated[
    Literal["auto", "low"],
    typer.Option(help="内容审核强度."),
]
BaseUrlField = Annotated[
    str,
    typer.Option(help="请求端点前缀."),
]


# ======================================================================================
# Helpers
# ======================================================================================
def _validate_api_key() -> str:
    """从环境变量读取 API Key, 未设置时抛出 BadParameter."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        errmsg = "未找到 API Key, 请设置环境变量 OPENAI_API_KEY"
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


def _create_temp_output(
    *,
    output_format: str,
) -> Path:
    """根据输出格式确定后缀名, 创建临时输出文件并返回其路径."""
    suffix = _FORMAT_SUFFIXES.get(output_format, f".{output_format}")
    with tempfile.NamedTemporaryFile(
        prefix="pymzckcli_t2i_",
        suffix=suffix,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    logger.info(
        "Created temporary output file: path=%r, format=%r", temp_path, output_format
    )
    return temp_path


def _guess_mime(
    *,
    path: Path,
) -> str:
    """根据文件后缀名猜测 MIME 类型, 未识别时默认 png."""
    return _MIME_TYPES.get(path.suffix.lower(), "image/png")


def _post_generation(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """调用图片生成接口(JSON), 返回响应 JSON."""
    logger.info(
        "Requesting image generation: model=%r, size=%r, quality=%r",
        payload.get("model"),
        payload.get("size"),
        payload.get("quality"),
    )
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code >= _HTTP_ERROR_THRESHOLD:
        errmsg = (
            f"Generation request failed: status={resp.status_code!r}, "
            f"body={resp.text!r}"
        )
        raise RuntimeError(errmsg)
    return resp.json()


def _post_edit(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    images: list[tuple[str, bytes, str]],
    mask: tuple[str, bytes, str] | None,
) -> dict[str, Any]:
    """调用图片编辑接口(multipart/form-data), 返回响应 JSON.

    NOTE: images/mask 元素为 (文件名, 内容, MIME 类型).
    """
    logger.info(
        "Requesting image edit: model=%r, size=%r, images=%d",
        payload.get("model"),
        payload.get("size"),
        len(images),
    )
    files: list[tuple[str, tuple[str, bytes, str]]] = [
        ("image", image) for image in images
    ]
    if mask is not None:
        files.append(("mask", mask))
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=payload,
        files=files,
        timeout=_REQUEST_TIMEOUT,
    )
    if resp.status_code >= _HTTP_ERROR_THRESHOLD:
        errmsg = f"Edit request failed: status={resp.status_code!r}, body={resp.text!r}"
        raise RuntimeError(errmsg)
    return resp.json()


def _extract_image(
    *,
    data: list[dict[str, Any]],
) -> bytes:
    """从响应 data 中提取第一张图片的二进制, 支持 b64_json 与 url 两种形式."""
    if not data:
        errmsg = "No image data received"
        raise RuntimeError(errmsg)
    if len(data) > 1:
        logger.warning(
            "Multiple images in response, only the first is saved: count=%d",
            len(data),
        )

    item = data[0]
    if b64 := item.get("b64_json"):
        return base64.b64decode(b64)
    if url := item.get("url"):
        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
        if resp.status_code >= _HTTP_ERROR_THRESHOLD:
            errmsg = (
                f"Failed to download image: url={url!r}, status={resp.status_code!r}"
            )
            raise RuntimeError(errmsg)
        return resp.content

    errmsg = "No image data found in response item: index=0"
    raise RuntimeError(errmsg)


def _save_image(
    *,
    image: bytes,
    output: str | None,
    output_format: str,
) -> Path:
    """保存图片, 未指定输出路径时创建临时文件."""
    if output is None:
        path = _create_temp_output(output_format=output_format)
    else:
        path = Path(output)

    path.write_bytes(image)
    typer.echo(f"图片已保存: {path} ({len(image)} bytes)")
    return path


# ======================================================================================
# Commands
# ======================================================================================
app = typer.Typer(
    name="openai",
    help="OpenAI Images API 兼容的图片生成/编辑工具",
    no_args_is_help=True,
)


@app.command()
def gen(  # noqa: PLR0913
    *,
    prompt: PromptField,
    prompt_type: PromptTypeField = _DEFAULT_PROMPT_TYPE,
    output: OutputField = None,
    model: ModelField = _DEFAULT_MODEL,
    size: SizeField = _DEFAULT_SIZE,
    quality: QualityField = _DEFAULT_QUALITY,
    output_format: OutputFormatField = _DEFAULT_OUTPUT_FORMAT,
    output_compression: OutputCompressionField = None,
    background: BackgroundField = _DEFAULT_BACKGROUND,
    moderation: ModerationField = _DEFAULT_MODERATION,
    base_url: BaseUrlField = _DEFAULT_BASE_URL,
) -> None:
    """图片生成: 通过提示词生成图片."""
    api_key = _validate_api_key()
    prompt = _resolve_prompt(prompt=prompt, prompt_type=prompt_type)

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
        "moderation": moderation,
    }
    if output_compression is not None:
        payload["output_compression"] = output_compression
    try:
        resp = _post_generation(
            url=f"{base_url}{_GENERATIONS_PATH}",
            api_key=api_key,
            payload=payload,
        )
        image = _extract_image(data=resp.get("data", []))
        _save_image(image=image, output=output, output_format=output_format)

    except Exception:
        logger.exception("Image generation failed")
        raise typer.Exit(code=1) from None


@app.command()
def edit(  # noqa: PLR0913
    *,
    prompt: PromptField,
    prompt_type: PromptTypeField = _DEFAULT_PROMPT_TYPE,
    image: ImageField,
    mask: MaskField = None,
    output: OutputField = None,
    model: ModelField = _DEFAULT_MODEL,
    size: SizeField = _DEFAULT_SIZE,
    quality: QualityField = _DEFAULT_QUALITY,
    input_fidelity: InputFidelityField = None,
    output_format: EditOutputFormatField = _DEFAULT_OUTPUT_FORMAT,
    output_compression: OutputCompressionField = None,
    background: EditBackgroundField = _DEFAULT_BACKGROUND,
    base_url: BaseUrlField = _DEFAULT_BASE_URL,
) -> None:
    """图片编辑: 基于原图按提示词修改, 支持遮罩指定可编辑区域."""
    api_key = _validate_api_key()
    prompt = _resolve_prompt(prompt=prompt, prompt_type=prompt_type)

    try:
        upload_images: list[tuple[str, bytes, str]] = []
        for path_text in image:
            image_path = Path(path_text)
            upload_images.append(
                (
                    image_path.name,
                    image_path.read_bytes(),
                    _guess_mime(path=image_path),
                )
            )
        upload_mask: tuple[str, bytes, str] | None = None
        if mask is not None:
            mask_path = Path(mask)
            upload_mask = (
                mask_path.name,
                mask_path.read_bytes(),
                _guess_mime(path=mask_path),
            )

    except OSError as e:
        errmsg = f"无法读取图片文件: {e}"
        raise typer.BadParameter(errmsg) from None

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
    }
    if input_fidelity is not None:
        payload["input_fidelity"] = input_fidelity
    if output_compression is not None:
        payload["output_compression"] = output_compression
    try:
        resp = _post_edit(
            url=f"{base_url}{_EDITS_PATH}",
            api_key=api_key,
            payload=payload,
            images=upload_images,
            mask=upload_mask,
        )
        image_data = _extract_image(data=resp.get("data", []))
        _save_image(image=image_data, output=output, output_format=output_format)

    except Exception:
        logger.exception("Image edit failed")
        raise typer.Exit(code=1) from None
