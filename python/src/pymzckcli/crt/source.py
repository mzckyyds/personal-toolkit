"""从图片或视频输入逐帧读取 RGB 图像."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

__all__ = [
    "Source",
    "is_image_path",
    "open_source",
]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
_RGB_CHANNELS = 3


@dataclass
class Source:
    """一个输入源: 类型/尺寸/时长 + 帧迭代器."""

    path: Path
    fps: float
    is_video: bool
    duration: float | None  # 视频秒数; 图片为 None (时长由调用方指定)
    width: int
    height: int
    _image: Image.Image | None = None

    def frames(self) -> Iterator[Image.Image]:
        """逐帧产出 RGB 图像; 图片无限重复, 视频播完即止."""
        if not self.is_video:
            img = self._image
            if img is None:
                errmsg = "图片源未加载"
                raise RuntimeError(errmsg)
            while True:
                yield img
        else:
            yield from self._video_frames()

    def _video_frames(self) -> Iterator[Image.Image]:
        cmd = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)  # noqa: S603
        if proc.stdout is None:
            errmsg = "无法打开 FFmpeg stdout"
            raise RuntimeError(errmsg)
        nbytes = self.width * self.height * _RGB_CHANNELS
        try:
            while True:
                buf = proc.stdout.read(nbytes)
                if len(buf) < nbytes:
                    break
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                    self.height, self.width, _RGB_CHANNELS
                )
                yield Image.fromarray(arr, mode="RGB")
        finally:
            proc.stdout.close()
            proc.wait()


def _frame_rate(stream: dict[str, str]) -> float:
    """解析 ffprobe 帧率 (num/den), 优先 r_frame_rate, 回退 avg_frame_rate."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        num, sep, den = stream.get(key, "").partition("/")
        if sep and float(den) > 0:
            rate = float(num) / float(den)
            if rate > 0:
                return rate
    errmsg = f"无法探测视频帧率: {stream!r}"
    raise ValueError(errmsg)


def open_source(*, path: Path, fps: float) -> Source:
    """按扩展名判定图片/视频; fps 仅用于图片, 视频用 ffprobe 探测尺寸/时长/帧率."""
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        return Source(
            path=path,
            fps=fps,
            is_video=False,
            duration=None,
            width=rgb.width,
            height=rgb.height,
            _image=rgb,
        )
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=True
    )
    info = json.loads(probe.stdout)
    stream = info["streams"][0]
    duration = float(info["format"]["duration"])
    return Source(
        path=path,
        fps=_frame_rate(stream),
        is_video=True,
        duration=duration,
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def is_image_path(path: Path) -> bool:
    """按扩展名判定是否为静态图片."""
    return path.suffix.lower() in _IMAGE_SUFFIXES
