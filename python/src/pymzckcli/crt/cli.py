"""CRT 雪花屏信号恢复生成器 CLI: image / video 两个子命令.

图片与视频输入的参数差异较大, 故拆分:
  image  必须指定 --duration (静态图恢复与保持的总时长), --fps 指定输出帧率.
  video  用 --intro 指定片头恢复跨度, 时长/帧率自动=输入, 播完即止.
--width/--height 缺省时沿用输入尺寸.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .pipeline import RenderConfig, render
from .source import is_image_path

__all__ = [
    "app",
]

app = typer.Typer(
    name="crt",
    help="CRT 雪花屏信号恢复 (NTSC/PAL 模拟电视仿真)",
    no_args_is_help=True,
)


# ======================================================================================
# Fields
# ======================================================================================
SourceArg = Annotated[
    Path,
    typer.Argument(
        help="输入图片/视频路径.", exists=True, dir_okay=False, readable=True
    ),
]
OutputOption = Annotated[
    Path,
    typer.Option("--output", "-o", help="输出 MP4 路径."),
]
DurationOption = Annotated[
    float,
    typer.Option("--duration", "-d", help="总时长 (秒), 图片必填.", min=0.5),
]
IntroOption = Annotated[
    float,
    typer.Option("--intro", help="片头恢复跨度 (秒).", min=0.5),
]
WidthOption = Annotated[
    int | None,
    typer.Option("--width", help="输出宽度 (像素), 缺省用输入宽度.", min=64),
]
HeightOption = Annotated[
    int | None,
    typer.Option("--height", help="输出高度 (像素), 缺省用输入高度.", min=64),
]
FpsOption = Annotated[
    float,
    typer.Option("--fps", help="输出帧率.", min=1.0),
]
StandardOption = Annotated[
    str,
    typer.Option("--standard", "-s", help="电视制式: ntsc 或 pal."),
]
SeedOption = Annotated[
    int,
    typer.Option("--seed", help="随机种子 (可复现)."),
]
SnowUntilOption = Annotated[
    float,
    typer.Option("--snow-until", help="纯雪花阶段占比 (0-1).", min=0.0, max=0.9),
]
LockByOption = Annotated[
    float,
    typer.Option("--lock-by", help="信号锁定时间点占比 (0-1).", min=0.1, max=1.0),
]
CrfOption = Annotated[
    int,
    typer.Option("--crf", help="x264 质量 (越小越清晰).", min=0, max=51),
]
PresetOption = Annotated[
    str,
    typer.Option("--preset", help="x264 编码预设."),
]
AudioOption = Annotated[
    bool,
    typer.Option("--audio/--no-audio", help="是否生成音频 (嘶声/原声)."),
]


# ======================================================================================
# Commands
# ======================================================================================
@app.command()
def image(  # noqa: PLR0913
    *,
    source: SourceArg,
    output: OutputOption,
    duration: DurationOption,
    width: WidthOption = None,
    height: HeightOption = None,
    fps: FpsOption = 30.0,
    standard: StandardOption = "ntsc",
    seed: SeedOption = 0,
    snow_until: SnowUntilOption = 0.16,
    lock_by: LockByOption = 0.62,
    crf: CrfOption = 17,
    preset: PresetOption = "slow",
    audio: AudioOption = True,
) -> None:
    """图片输入: 生成从雪花恢复为清晰图片并保持的视频 (需 --duration)."""
    if not is_image_path(source):
        errmsg = "image 命令仅接受图片输入, 视频请用 `crt video`."
        raise typer.BadParameter(errmsg)
    config = RenderConfig(
        image=source,
        output=output,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        standard=standard,
        seed=seed,
        snow_until=snow_until,
        lock_by=lock_by,
        crf=crf,
        preset=preset,
        audio=audio,
    )
    render(cfg=config)


@app.command()
def video(  # noqa: PLR0913
    *,
    source: SourceArg,
    output: OutputOption,
    intro: IntroOption = 4.0,
    width: WidthOption = None,
    height: HeightOption = None,
    standard: StandardOption = "ntsc",
    seed: SeedOption = 0,
    snow_until: SnowUntilOption = 0.16,
    lock_by: LockByOption = 0.62,
    crf: CrfOption = 17,
    preset: PresetOption = "slow",
    audio: AudioOption = True,
) -> None:
    """视频输入: 片头雪花恢复后播放原片, 输入播完即停止 (时长/帧率自动)."""
    if is_image_path(source):
        errmsg = "video 命令仅接受视频输入, 图片请用 `crt image`."
        raise typer.BadParameter(errmsg)
    config = RenderConfig(
        image=source,
        output=output,
        intro=intro,
        width=width,
        height=height,
        standard=standard,
        seed=seed,
        snow_until=snow_until,
        lock_by=lock_by,
        crf=crf,
        preset=preset,
        audio=audio,
    )
    render(cfg=config)
