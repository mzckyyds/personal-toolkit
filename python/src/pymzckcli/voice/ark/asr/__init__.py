"""火山引擎单向流式语音识别工具(基于 WebSocket).

密钥优先使用环境变量 ARK_ASR_API_KEY, 其次使用 ARK_API_KEY.

reference: https://docs.volcengine.com/docs/6561/2628951?lang=zh
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from . import _protocol

__all__ = [
    "recognize",
]


logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/plan/sauc/bigmodel_nostream"
_DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"
_DEFAULT_SEG_DURATION = 200
_DEFAULT_AUDIO_LANGUAGE = None


# ======================================================================================
# Fields
# ======================================================================================
FileField = Annotated[
    str,
    typer.Option(help="音频文件路径."),
]
OutputField = Annotated[
    str | None,
    typer.Option(help="识别结果保存路径, 未指定时仅输出到终端."),
]
EndpointField = Annotated[
    str,
    typer.Option(help="请求端点(WebSocket)."),
]
ResourceIdField = Annotated[
    str,
    typer.Option(help="请求的模型版本."),
]
SegDurationField = Annotated[
    int,
    typer.Option(help="每个音频包时长 (ms)."),
]
AudioLanguageField = Annotated[
    str | None,
    typer.Option(
        help="为空时支持识别以下语种: 中文/英文/上海话/闽南话/四川话/陕西话/粤语."
    ),
]


# ======================================================================================
# Commands
# ======================================================================================
def recognize(  # noqa: PLR0913
    *,
    file: FileField,
    output: OutputField = None,
    endpoint: EndpointField = _DEFAULT_ENDPOINT,
    resource_id: ResourceIdField = _DEFAULT_RESOURCE_ID,
    seg_duration: SegDurationField = _DEFAULT_SEG_DURATION,
    audio_language: AudioLanguageField = _DEFAULT_AUDIO_LANGUAGE,
) -> None:
    """识别音频文件并输出转写文本."""
    api_key = os.environ.get("ARK_ASR_API_KEY") or os.environ.get("ARK_API_KEY")
    if api_key is None:
        errmsg = "未找到 API Key, 请设置环境变量 ARK_ASR_API_KEY 或 ARK_API_KEY"
        raise typer.BadParameter(errmsg)

    config = _protocol.Config(
        api_key=api_key,
        resource_id=resource_id,
    )
    # `_protocol.AsrWsClient` 会将音频文件转换为如下格式
    audio: dict[str, Any] = {
        "format": "wav",
        "codec": "raw",
        "rate": _protocol.DEFAULT_SAMPLE_RATE,
        "bits": 16,
        "channel": 1,
    }
    if audio_language is not None:
        audio["language"] = audio_language
    payload: dict[str, Any] = {
        "audio": audio,
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_ddc": True,
            "show_utterances": False,
        },
    }

    async def _run() -> str:
        final_text = ""

        async with _protocol.AsrWsClient(endpoint, seg_duration) as client:
            async for response in client.execute(file, config, payload):  # pyright: ignore[reportUnknownMemberType]
                if response.payload_msg is not None:
                    result = response.payload_msg.get("result", {})
                    if text := result.get("text"):
                        final_text += text

        return final_text

    try:
        final_text = asyncio.run(_run())
    except Exception:
        logger.exception("ASR processing failed")
        raise typer.Exit(code=1) from None

    if not final_text:
        logger.warning("未识别到文本, 请检查输入音频与 audio_* 参数是否匹配.")
        return

    typer.echo(f"识别结果: {final_text}")

    if output is not None:
        Path(output).write_text(final_text, encoding="utf-8")
        typer.echo(f"识别结果已保存: {output}")
