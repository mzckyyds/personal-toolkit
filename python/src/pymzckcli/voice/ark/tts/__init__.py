"""火山引擎单向流式语音合成工具(基于 WebSocket).

密钥优先使用环境变量 ARK_TTS_API_KEY, 其次使用 ARK_API_KEY.

reference: https://docs.volcengine.com/docs/6561/2534913?lang=zh
"""

import asyncio
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

import typer
import websockets

from . import _protocol

__all__ = [
    "synthesize",
]


logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = (
    "wss://openspeech.bytedance.com/api/v3/plan/tts/unidirectional/stream"
)
_DEFAULT_RESOURCE_ID = "seed-tts-2.0"
_DEFAULT_SPEAKER = "zh_male_liufei_uranus_bigtts"
_DEFAULT_AUDIO_FORMAT = "mp3"
_DEFAULT_AUDIO_SAMPLE_RATE = None
_DEFAULT_AUDIO_BIT_RATE = None
_DEFAULT_AUDIO_SPEECH_RATE = 0
_DEFAULT_AUDIO_LOUDNESS_RATE = 0

# 音频格式 -> 临时文件后缀名
_FORMAT_SUFFIXES = {
    "mp3": ".mp3",
    "pcm": ".pcm",
    "ogg_opus": ".ogg",
    "wav": ".wav",
}


# ======================================================================================
# Fields
# ======================================================================================
TextField = Annotated[
    str | None,
    typer.Option(help="待合成文本, 与 file 二选一."),
]
FileField = Annotated[
    str | None,
    typer.Option(help="待合成文本的文件路径, 与 text 二选一."),
]
OutputField = Annotated[
    str | None,
    typer.Option(help="音频保存路径, 未指定时自动创建临时文件."),
]
TimestampField = Annotated[
    str | None,
    typer.Option(help="字级时间戳保存路径, 指定时将自动启用接口的字幕服务."),
]
ContextTextsField = Annotated[
    list[str],
    typer.Option(
        "--context-text",
        help="控制说话语调和情绪的上下文文本, 可多次指定, "
        "序列化后传入 req_params.additions.",
    ),
]
EndpointField = Annotated[
    str,
    typer.Option(help="请求端点(WebSocket)."),
]
ResourceIdField = Annotated[
    str,
    typer.Option(help="请求的模型版本."),
]
SpeakerField = Annotated[
    str,
    typer.Option(help="音色."),
]
AudioFormatField = Annotated[
    str,
    typer.Option(help="指定音频格式: mp3/pcm/ogg_opus/wav."),
]
AudioSampleRateField = Annotated[
    int | None,
    typer.Option(
        help="指定输出音频的采样率(Hz), 不同音频格式支持的采样率不同, "
        "wav/pcm/mp3 默认值为24000, 可选值为 [8000, 16000, 22050, 24000, "
        "32000, 44100, 48000], ogg_opus 仅支持 48k."
    ),
]
AudioBitRateField = Annotated[
    int | None,
    typer.Option(
        help="指定音频比特率(bps), mp3 默认值为 64000, 可选值为 [64000, 160000], "
        "ogg_opus 可选值为 [64000, 160000], wav/pcm 不支持."
    ),
]
AudioSpeechRateField = Annotated[
    int,
    typer.Option(
        help="指定音频的语速, 默认值为0, 取值范围 [-50, 100]. "
        "取值 100 代表 2.0 倍速, -50 代表0.5倍速."
    ),
]
AudioLoudnessRateField = Annotated[
    int,
    typer.Option(
        help="指定音频的音量, 默认值为0, 取值范围 [-50, 100]. "
        "取值 100 代表 2.0 倍音量, -50 代表 0.5 倍音量."
    ),
]


# ======================================================================================
# Helpers
# ======================================================================================
def _create_temp_output(
    *,
    audio_format: str,
) -> Path:
    """根据音频格式确定后缀名, 创建临时输出文件并返回其路径."""
    suffix = _FORMAT_SUFFIXES.get(audio_format, f".{audio_format}")
    with tempfile.NamedTemporaryFile(
        prefix="pymzckcli_tts_",
        suffix=suffix,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
    logger.info(
        "Created temporary output file: path=%r, format=%r", temp_path, audio_format
    )
    return temp_path


def _parse_subtitle(
    *,
    payload: bytes,
) -> dict[str, Any] | None:
    """解析 TTSSubtitle 事件 payload, 失败时返回 None."""
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse subtitle payload: %r", e)
        return None


def _accumulate(
    *,
    msg: _protocol.Message,
    audio_data: bytearray,
    subtitles: list[dict[str, Any]],
) -> None:
    """将服务端消息归集为音频与字幕数据.

    NOTE: 字级时间戳在 TTSSubtitle 事件中, 每个句子一条,
    TTSSentenceStart/End 中的 words 恒为空.
    """
    if msg.type == _protocol.MsgType.AudioOnlyServer and msg.payload:
        audio_data.extend(msg.payload)
    elif (
        msg.type == _protocol.MsgType.FullServerResponse
        and msg.event == _protocol.EventType.TTSSubtitle
        and msg.payload
        and (subtitle := _parse_subtitle(payload=msg.payload)) is not None
    ):
        subtitles.append(subtitle)


async def _synthesize(  # noqa: PLR0913
    *,
    url: str,
    api_key: str,
    resource_id: str,
    speaker: str,
    text: str,
    context_texts: list[str],
    audio_format: str,
    audio_sample_rate: int | None,
    audio_bit_rate: int | None,
    audio_speech_rate: int,
    audio_loudness_rate: int,
    audio_enable_subtitle: bool,
) -> tuple[bytes, list[dict[str, Any]]]:
    """连接 TTS WebSocket 单向流接口, 返回合成音频数据与字幕数据."""
    headers = {
        "X-Api-Key": api_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Control-Require-Usage-Tokens-Return": "*",
    }
    audio_params: dict[str, Any] = {
        "format": audio_format,
        "speech_rate": audio_speech_rate,
        "loudness_rate": audio_loudness_rate,
        "enable_subtitle": audio_enable_subtitle,
    }
    if audio_sample_rate is not None:
        audio_params["sample_rate"] = audio_sample_rate
    if audio_bit_rate is not None:
        audio_params["bit_rate"] = audio_bit_rate

    req_params: dict[str, Any] = {
        "speaker": speaker,
        "text": text,
        "audio_params": audio_params,
    }
    if context_texts:
        # NOTE: additions 为 JSON 字符串而非对象
        req_params["additions"] = json.dumps({"context_texts": context_texts})

    body: dict[str, Any] = {"req_params": req_params}

    websocket = await websockets.connect(
        url,
        additional_headers=headers,
        max_size=10 * 1024 * 1024,
    )
    if websocket.response is None:
        errmsg = "Failed to get response from WebSocket server"
        raise RuntimeError(errmsg)
    logger.info(
        "Connected to WebSocket server, Logid: %r",
        websocket.response.headers.get("x-tt-logid", ""),
    )

    try:
        await _protocol.full_client_request(websocket, json.dumps(body).encode())  # pyright: ignore[reportUnknownMemberType]

        audio_data = bytearray()
        subtitles: list[dict[str, Any]] = []
        while True:
            msg = await _protocol.receive_message(websocket)  # pyright: ignore[reportUnknownMemberType]

            if msg.type == _protocol.MsgType.Error:
                errmsg = f"TTS conversion failed: {msg}"
                raise RuntimeError(errmsg)
            if (
                msg.type == _protocol.MsgType.FullServerResponse
                and msg.event == _protocol.EventType.SessionFinished
            ):
                break
            _accumulate(msg=msg, audio_data=audio_data, subtitles=subtitles)

        if not audio_data:
            errmsg = "No audio data received"
            raise RuntimeError(errmsg)

        logger.info(
            "Audio received: %d, subtitles: %d", len(audio_data), len(subtitles)
        )
        return bytes(audio_data), subtitles
    finally:
        await websocket.close()
        logger.info("Connection closed")


# ======================================================================================
# Commands
# ======================================================================================
def synthesize(  # noqa: PLR0913
    *,
    text: TextField = None,
    file: FileField = None,
    output: OutputField = None,
    timestamp: TimestampField = None,
    context_texts: ContextTextsField = [],  # noqa: B006
    endpoint: EndpointField = _DEFAULT_ENDPOINT,
    resource_id: ResourceIdField = _DEFAULT_RESOURCE_ID,
    speaker: SpeakerField = _DEFAULT_SPEAKER,
    audio_format: AudioFormatField = _DEFAULT_AUDIO_FORMAT,
    audio_sample_rate: AudioSampleRateField = _DEFAULT_AUDIO_SAMPLE_RATE,
    audio_bit_rate: AudioBitRateField = _DEFAULT_AUDIO_BIT_RATE,
    audio_speech_rate: AudioSpeechRateField = _DEFAULT_AUDIO_SPEECH_RATE,
    audio_loudness_rate: AudioLoudnessRateField = _DEFAULT_AUDIO_LOUDNESS_RATE,
) -> None:
    """合成语音并保存到文件."""
    api_key = os.environ.get("ARK_TTS_API_KEY") or os.environ.get("ARK_API_KEY")
    if api_key is None:
        errmsg = "未找到 API Key, 请设置环境变量 ARK_TTS_API_KEY 或 ARK_API_KEY"
        raise typer.BadParameter(errmsg)

    if text is not None and file is not None:
        errmsg = "text 与 file 参数只能二选一"
        raise typer.BadParameter(errmsg)

    if file is not None:
        try:
            text = Path(file).read_text(encoding="utf-8")
        except OSError as e:
            errmsg = f"无法读取文本文件: {file!r} ({e})"
            raise typer.BadParameter(errmsg) from None

    if text is None:
        errmsg = "text 与 file 参数必须指定其一"
        raise typer.BadParameter(errmsg)

    try:
        audio_data, subtitles = asyncio.run(
            _synthesize(
                url=endpoint,
                api_key=api_key,
                resource_id=resource_id,
                speaker=speaker,
                text=text,
                context_texts=context_texts,
                audio_format=audio_format,
                audio_sample_rate=audio_sample_rate,
                audio_bit_rate=audio_bit_rate,
                audio_speech_rate=audio_speech_rate,
                audio_loudness_rate=audio_loudness_rate,
                audio_enable_subtitle=bool(timestamp),
            )
        )

    except Exception:
        logger.exception("TTS synthesis failed")
        raise typer.Exit(code=1) from None

    if output is None:
        output = str(_create_temp_output(audio_format=audio_format))

    Path(output).write_bytes(audio_data)
    typer.echo(f"音频已保存: {output} ({len(audio_data)} bytes)")

    if timestamp:
        Path(timestamp).write_text(
            json.dumps(subtitles, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        typer.echo(f"字级时间戳已保存: {timestamp}")
