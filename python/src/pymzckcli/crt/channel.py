"""信道模型: 射频噪声, 多径重影, 慢衰落与同频干扰.

把发射机输出的复合视频当作经过射频链路后的接收信号. 这里不做简单的"加噪声",
而是按模拟电视接收的物理机制建模:

* 热噪声/散粒噪声 -> 加性高斯白噪声. 弱信号时噪声淹没同步头, 屏幕呈雪花,
  这正是"无信号"状态的来源.
* 多径传播 -> 延迟回波 (重影/ghost), 表现为画面右侧的淡影.
* 信号衰落 -> 慢速幅度起伏 (flutter), 弱信号时画面"呼吸".
* 同频干扰 -> 一个偏频正弦, 产生网纹 (herringbone).

噪声是逐帧连续处理的 (白噪声无记忆), 重影通过跨帧延迟缓冲保持连续.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .standards import Standard

__all__ = [
    "Channel",
    "ChannelParams",
]

_AMP_EPS = 1e-4  # 判定重影/衰落/干扰是否启用的幅度阈值
_NOISE_EPS = 1e-6  # 判定是否叠加热噪声的阈值
_MAX_GHOST_DELAY_US = 8.0  # 重影延迟缓冲上限
_SIGNAL_REF_CNR_DB = 34.0  # 信号幅度达到满幅时的载噪比
_THERMAL_NOISE_SIGMA = 0.05  # 固定热噪声 (相对满幅同步头)


@dataclass
class ChannelParams:
    """某一时刻的信道状态 (由时间线插值得到)."""

    cnr_db: float = 0.0  # 载噪比, 越小信号越弱
    ghost_amp: float = 0.0  # 主重影幅度 (相对信号)
    ghost_delay_us: float = 1.5  # 主重影延迟 (µs)
    ghost2_amp: float = 0.0  # 次重影幅度
    ghost2_delay_us: float = 4.0
    fade_depth: float = 0.0  # 慢衰落深度 (0~1)
    fade_rate_hz: float = 0.7  # 衰落频率
    interference_amp: float = 0.0  # 同频干扰幅度
    interference_beat_hz: float = 15_000.0  # 干扰偏频 (产生网纹)


@dataclass
class Channel:
    """流式信道. 跨帧保存重影延迟缓冲与衰落相位."""

    std: Standard
    seed: int = 0
    _rng: np.random.Generator = field(init=False)
    _tail: np.ndarray = field(init=False)
    _t: float = 0.0  # 已处理的绝对时间 (s)
    _fade_phase: float = 0.0

    def __post_init__(self) -> None:
        """初始化随机源与重影延迟缓冲."""
        self._rng = np.random.default_rng(self.seed)
        n = round(_MAX_GHOST_DELAY_US * 1e-6 * self.std.fs) + 8
        self._tail = np.zeros(n, dtype=np.float64)

    def process(self, *, composite: np.ndarray, p: ChannelParams) -> np.ndarray:
        """对一帧复合信号施加信道损伤, 返回接收信号 (float64)."""
        fs = self.std.fs
        x = np.asarray(composite, dtype=np.float64)
        n = x.size

        # 1) 多径重影: 用延迟缓冲拼接, 保证跨帧连续.
        if p.ghost_amp > _AMP_EPS or p.ghost2_amp > _AMP_EPS:
            buf = np.concatenate([self._tail, x])
            out = x.copy()
            for amp, delay_us in (
                (p.ghost_amp, p.ghost_delay_us),
                (p.ghost2_amp, p.ghost2_delay_us),
            ):
                if amp <= _AMP_EPS:
                    continue
                d = round(delay_us * 1e-6 * fs)
                if 0 < d < buf.size:
                    out += amp * buf[buf.size - n - d : buf.size - d]
            x = out
            self._tail = buf[-self._tail.size :].copy()
        else:
            self._tail = np.concatenate([self._tail, x])[-self._tail.size :].copy()

        # 2) 慢衰落 (幅度呼吸).
        if p.fade_depth > _AMP_EPS:
            t = self._t + np.arange(n) / fs
            fade = 1.0 + p.fade_depth * np.sin(
                2 * np.pi * p.fade_rate_hz * t + self._fade_phase
            )
            x = x * fade

        # 3) 同频干扰网纹.
        if p.interference_amp > _AMP_EPS:
            t = self._t + np.arange(n) / fs
            x = x + p.interference_amp * np.sin(2 * np.pi * p.interference_beat_hz * t)

        # 4) 信号幅度随强度衰减 (弱信号 = 弱载波), 再叠加固定热噪声.
        #    AGC 在弱信号时提升增益, 同时放大噪声 -> 画面从雪花中逐渐浮现.
        signal_amp = float(
            np.clip(10.0 ** ((p.cnr_db - _SIGNAL_REF_CNR_DB) / 20.0), 0.0, 1.0)
        )
        x = x * signal_amp
        if _THERMAL_NOISE_SIGMA > _NOISE_EPS:
            x = x + self._rng.normal(0.0, _THERMAL_NOISE_SIGMA, n)

        self._t += n / fs
        return x
