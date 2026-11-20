"""CRT 信号恢复仿真与 MP4 渲染管线."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .audio import generate_hiss
from .channel import Channel
from .display import CRTDisplay
from .encoder import Encoder
from .receiver import Receiver
from .source import open_source
from .standards import get_standard
from .timeline import RecoveryTimeline

if TYPE_CHECKING:
    from collections.abc import Iterator

    from PIL import Image

    from .standards import Standard

logger = logging.getLogger(__name__)

__all__ = [
    "RenderConfig",
    "render",
]


@dataclass
class RenderConfig:
    """渲染参数."""

    image: Path
    output: Path
    duration: float = 12.0
    width: int | None = None  # None = 沿用输入尺寸
    height: int | None = None
    fps: float = 30.0
    standard: str = "ntsc"
    seed: int = 0
    snow_until: float = 0.16
    lock_by: float = 0.62
    intro: float = 4.0  # 视频输入的片头恢复跨度 (秒)
    crf: int = 17
    preset: str = "slow"
    audio: bool = True


def _ffmpeg_command(*, cfg: RenderConfig, target: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{cfg.width}x{cfg.height}",
        "-r",
        f"{cfg.fps}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(cfg.crf),
        "-preset",
        cfg.preset,
        "-movflags",
        "+faststart",
        str(target),
    ]


def _encode_video(  # noqa: PLR0913
    *,
    cfg: RenderConfig,
    target: Path,
    std: Standard,
    encoder: Encoder,
    channel: Channel,
    receiver: Receiver,
    display: CRTDisplay,
    timeline: RecoveryTimeline,
    frames: Iterator[Image.Image],
    first: Image.Image,
) -> list[tuple[float, float]]:
    """逐场仿真并把输出帧写入 FFmpeg, 返回逐帧 (时间, 噪声电平)."""
    n_out = max(1, round(cfg.duration * cfg.fps))
    dt_out = cfg.duration / n_out
    field_period = 1.0 / std.field_rate
    max_frames = int(np.ceil(cfg.duration * std.field_rate / 2.0)) + 3

    cmd = _ffmpeg_command(cfg=cfg, target=target)
    logger.info("启动 FFmpeg: %r", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)  # noqa: S603
    if proc.stdin is None:
        errmsg = "无法打开 FFmpeg stdin"
        raise RuntimeError(errmsg)

    hiss_levels: list[tuple[float, float]] = []
    t_sim = 0.0
    out_idx = 0
    frame_idx = 0
    current = first
    try:
        while out_idx < n_out and frame_idx < max_frames:
            encoder.set_image(current)
            ctl = timeline.controls_at(t=t_sim)
            receiver.h_hold_error = ctl.h_hold
            receiver.v_hold_error = ctl.v_hold
            receiver.tune_offset = ctl.tune
            params = timeline.params_at(t=t_sim)
            composite = encoder.encode_frame()
            rx = channel.process(composite=composite, p=params)
            fields = receiver.process_frame(composite=rx)
            for field_index, fd in enumerate(fields):
                display.paint_field(fd=fd, field_index=field_index)
                t_sim += field_period
                while out_idx < n_out and t_sim >= (out_idx + 1) * dt_out:
                    frame = display.snapshot()
                    proc.stdin.write(frame.tobytes())
                    hiss_levels.append(((out_idx + 0.5) * dt_out, receiver.noise_sigma))
                    out_idx += 1
                    if out_idx % 30 == 0:
                        logger.info("进度: %d/%d 帧", out_idx, n_out)
            frame_idx += 1
            nxt = next(frames, None)
            if nxt is None:
                break  # 视频输入播完即止
            current = nxt
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        errmsg = f"FFmpeg 退出码 {proc.returncode}"
        raise RuntimeError(errmsg)
    return hiss_levels


def _mux_audio(*, video: Path, audio: Path, output: Path) -> None:
    """把嘶声音频混入视频 (视频流直接拷贝)."""
    mux = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output),
    ]
    logger.info("混入音频: %r", " ".join(mux))
    result = subprocess.run(mux, check=False)  # noqa: S603
    if result.returncode != 0:
        errmsg = f"音频混流 FFmpeg 退出码 {result.returncode}"
        raise RuntimeError(errmsg)


def render(*, cfg: RenderConfig) -> Path:
    """渲染 CRT 信号恢复视频, 返回输出文件路径."""
    std = get_standard(cfg.standard)
    src = open_source(path=cfg.image, fps=cfg.fps)
    if src.is_video:
        if src.duration is None or src.duration <= 0:
            errmsg = f"无法探测视频时长: {cfg.image}"
            raise ValueError(errmsg)
        cfg.duration = src.duration
        cfg.fps = src.fps
        span = min(cfg.intro, cfg.duration)
    else:
        span = cfg.duration
    if cfg.width is None:
        cfg.width = src.width
    if cfg.height is None:
        cfg.height = src.height
    frames = src.frames()
    first = next(frames)
    encoder = Encoder(std=std, image=first)
    channel = Channel(std=std, seed=cfg.seed)
    receiver = Receiver(std=std, seed=cfg.seed)
    display = CRTDisplay(std=std, width=cfg.width, height=cfg.height, seed=cfg.seed)
    timeline = RecoveryTimeline(
        duration=span,
        seed=cfg.seed,
        snow_until=cfg.snow_until,
        lock_by=cfg.lock_by,
    )
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    if cfg.audio:
        with tempfile.TemporaryDirectory(prefix="crt_") as tmp:
            video_target = Path(tmp) / "video.mp4"
            levels = _encode_video(
                cfg=cfg,
                target=video_target,
                std=std,
                encoder=encoder,
                channel=channel,
                receiver=receiver,
                display=display,
                timeline=timeline,
                frames=frames,
                first=first,
            )
            audio_path = Path(tmp) / "hiss.wav"
            generate_hiss(
                levels=levels,
                duration=cfg.duration,
                path=audio_path,
                seed=cfg.seed,
                input_audio=cfg.image if src.is_video else None,
            )
            _mux_audio(video=video_target, audio=audio_path, output=cfg.output)
    else:
        _encode_video(
            cfg=cfg,
            target=cfg.output,
            std=std,
            encoder=encoder,
            channel=channel,
            receiver=receiver,
            display=display,
            timeline=timeline,
            frames=frames,
            first=first,
        )
    logger.info("完成: %r", cfg.output)
    return cfg.output
