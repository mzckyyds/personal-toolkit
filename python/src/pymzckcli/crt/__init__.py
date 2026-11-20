"""CRT 雪花屏信号恢复生成器 (NTSC/PAL 模拟电视仿真)."""

from .cli import app
from .pipeline import RenderConfig, render

__all__ = [
    "RenderConfig",
    "app",
    "render",
]
