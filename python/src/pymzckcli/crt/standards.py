"""NTSC/PAL 模拟电视制式常量与复合信号波形定义.

所有时间单位为秒, 电平以消隐电平为 0, 白电平为 1 归一化, 同步头为负值.
采样率取 4 倍色副载波 (4fsc), 这是数字解码 NTSC/PAL 的经典采样方式.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SYNC_LEVEL = -0.40  # 同步头相对消隐电平 (NTSC 40 IRE / 100 IRE)
BLACK_LEVEL_NTSC = 0.075  # NTSC 7.5 IRE 黑电平抬升 (setup)
BURST_AMP = 0.20  # 色同步突发幅度 (±20 IRE)


@dataclass(frozen=True)
class Standard:
    """一种模拟电视制式的物理参数."""

    name: str
    total_lines: int  # 每帧总行数
    field_lines: float  # 每场行数 (含半行, 产生隔行)
    active_lines: int  # 每帧有效行数
    line_rate: float  # 行频 Hz
    field_rate: float  # 场频 Hz
    fsc: float  # 色副载波 Hz
    sync_us: float  # 行同步宽度
    eq_us: float  # 均衡脉冲宽度
    blanking_us: float  # 行消隐总宽
    active_start_us: float  # 有效视频起点 (相对行首)
    active_us: float  # 有效视频宽度
    burst_start_us: float  # 色同步突发起点 (相对行首)
    burst_cycles: float  # 突发周期数
    black_level: float  # 黑电平
    pal: bool  # 是否 PAL (V 分量逐行倒相)

    @property
    def fs(self) -> float:
        """采样率 = 4fsc."""
        return 4.0 * self.fsc

    @property
    def samples_per_line(self) -> float:
        """每行采样点数."""
        return self.fs / self.line_rate

    @property
    def line_period(self) -> float:
        """行周期 (秒)."""
        return 1.0 / self.line_rate

    @property
    def field_period(self) -> float:
        """场周期 (秒)."""
        return 1.0 / self.field_rate

    @property
    def active_lines_per_field(self) -> int:
        """每场有效行数."""
        return self.active_lines // 2

    @property
    def active_width(self) -> int:
        """有效视频水平像素数 (取整到 16 的倍数)."""
        n = int(self.active_us * 1e-6 * self.fs)
        return (n // 16) * 16


NTSC = Standard(
    name="ntsc",
    total_lines=525,
    field_lines=262.5,
    active_lines=480,
    line_rate=15734.265734,
    field_rate=59.94005994,
    fsc=315.0 / 88.0 * 1e6,
    sync_us=4.7,
    eq_us=2.35,
    blanking_us=10.9,
    active_start_us=10.9,
    active_us=52.2,
    burst_start_us=5.3,
    burst_cycles=9.0,
    black_level=BLACK_LEVEL_NTSC,
    pal=False,
)

PAL = Standard(
    name="pal",
    total_lines=625,
    field_lines=312.5,
    active_lines=576,
    line_rate=15625.0,
    field_rate=50.0,
    fsc=4433618.75,
    sync_us=4.7,
    eq_us=2.35,
    blanking_us=12.0,
    active_start_us=12.0,
    active_us=51.9,
    burst_start_us=5.6,
    burst_cycles=10.0,
    black_level=0.0,
    pal=True,
)

STANDARDS = {"ntsc": NTSC, "pal": PAL}


def get_standard(name: str) -> Standard:
    """按名称查表返回制式, 未知名称抛 ValueError."""
    key = name.lower()
    if key not in STANDARDS:
        msg = f"未知制式 {name!r}, 可选: {sorted(STANDARDS)}"
        raise ValueError(msg)
    return STANDARDS[key]


# 每场垂直同步块结构 (相对场边界, 单位: 行):
#   NTSC 前置均衡 3 + 开槽垂直同步 3 + 后置均衡 3 = 9 行
#   PAL  2.5 + 2.5 + 2.5 = 7.5 行
def _block_widths(std: Standard) -> tuple[float, float, float]:
    w = 2.5 if std.pal else 3.0
    return w, w, w


def line_kind(std: Standard, line_index: int) -> tuple[str, int]:
    """判定某整行在帧内的类型与有效行号.

    返回 (kind, row): kind ∈ {"pre_eq","serr","post_eq","blank","active"},
    row 为有效行在场内的序号 (仅 active 有意义).
    场边界位于半行精度 (field_lines 含 .5), 由此自然产生隔行.
    """
    fl = std.field_lines
    field = int((line_index + 0.5) // fl)
    rel = line_index - field * fl  # 相对场起始的行位置 (边界行可能为 -0.5)
    pre_w, serr_w, post_w = _block_widths(std)
    block_total = pre_w + serr_w + post_w
    if 0 <= rel < pre_w:
        return "pre_eq", -1
    if pre_w <= rel < pre_w + serr_w:
        return "serr", -1
    if pre_w + serr_w <= rel < block_total:
        return "post_eq", -1
    start = _active_start_line(std)
    n = std.active_lines_per_field
    if start <= rel < start + n:
        return "active", int(rel - start)
    return "blank", -1


def _active_start_line(std: Standard) -> float:
    """有效视频区相对场边界的起始行."""
    return 17.0 if std.pal else 20.0


def sync_pulse(t_local: np.ndarray, std: Standard) -> np.ndarray:
    """普通行的行同步波形 (t_local 为相对行首的时间, 秒)."""
    return np.where(t_local < std.sync_us * 1e-6, SYNC_LEVEL, 0.0)


def equalizing_pulse(t_local: np.ndarray, std: Standard) -> np.ndarray:
    """均衡脉冲行: 每半行一个窄脉冲."""
    half = std.line_period / 2.0
    w = std.eq_us * 1e-6
    t0 = np.mod(t_local, half)
    return np.where(t0 < w, SYNC_LEVEL, 0.0)


def serration_pulse(t_local: np.ndarray, std: Standard) -> np.ndarray:
    """开槽垂直同步行: 大部分为低电平, 每半行开头一个行同步宽度的高槽."""
    half = std.line_period / 2.0
    t0 = np.mod(t_local, half)
    return np.where(t0 < std.sync_us * 1e-6, 0.0, SYNC_LEVEL)
