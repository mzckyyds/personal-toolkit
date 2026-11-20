"""发射端: 把静态图像编码为连续的 NTSC/PAL 复合视频信号.

复合信号 = 行/场同步 + 消隐 + 色同步突发 + 亮度 + 正交平衡调幅色度.
采样率 4fsc, 副载波相位随全局采样索引连续累积 (保留 NTSC 4 场 / PAL 8 场序列).
"""

# numpy 数组运算的类型存根返回 Any 形状, 以下两项 unknown 检查对纯数值数组代码不适用
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from .standards import (
    BURST_AMP,
    equalizing_pulse,
    line_kind,
    serration_pulse,
    sync_pulse,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .standards import Standard

__all__ = [
    "Encoder",
]

# 色度带宽限制 (Hz): 模拟电视色度通道远窄于亮度
_CHROMA_BW = 1.3e6
_US = 1e-6  # 微秒 -> 秒


def _limit_chroma_bandwidth(
    sig: NDArray[np.float64], fs: float, bw: float
) -> NDArray[np.float64]:
    """对色度做带限 (高斯低通近似), 模拟色度通道窄带."""
    sigma = fs / (2.0 * np.pi * bw)
    return gaussian_filter1d(sig, sigma, mode="nearest")


class Encoder:
    """把一张图像编码为逐帧复合信号流."""

    def __init__(self, *, std: Standard, image: Image.Image) -> None:
        """按制式有效分辨率拟合图像, 并预计算亮度/色度平面."""
        self.std = std
        self.width = std.active_width
        self.height = std.active_lines
        self._line_cursor = 0  # 全局行计数
        self.set_image(image)

    def set_image(self, image: Image.Image) -> None:
        """更换输入图像 (视频逐帧), 重算亮度/色度平面, 不重置行计数."""
        std = self.std
        rgb = _fit_image(image=image, width=self.width, height=self.height)
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        # 黑电平抬升 (NTSC setup)
        y = std.black_level + y * (1.0 - std.black_level)
        u = 0.492 * (b - y)
        v = 0.877 * (r - y)
        # 色度带限, 像素域采样率取宽除以有效时长
        self.y = y
        px_fs = self.width / (std.active_us * _US)
        self.u = _limit_chroma_bandwidth(u, px_fs, _CHROMA_BW)
        self.v = _limit_chroma_bandwidth(v, px_fs, _CHROMA_BW)

    def encode_frame(self) -> NDArray[np.float64]:
        """生成下一帧的复合信号 (一维 float64)."""
        std = self.std
        spl = std.samples_per_line
        n_lines = std.total_lines
        start = round(self._line_cursor * spl)
        end = round((self._line_cursor + n_lines) * spl)
        idx = np.arange(start, end)
        t = idx / std.fs
        out = np.zeros(idx.size, dtype=np.float64)

        # 逐行填充
        line_starts = np.round((self._line_cursor + np.arange(n_lines)) * spl).astype(
            np.int64
        )
        for k in range(n_lines):
            lg = self._line_cursor + k
            s0 = line_starts[k] - start
            s1 = (line_starts[k + 1] - start) if k + 1 < n_lines else idx.size
            if s1 <= s0:
                continue
            tl = t[s0:s1] - lg * std.line_period
            kind, row = line_kind(std, lg)
            seg = self._line_waveform(tl, t[s0:s1], kind, row, lg)
            out[s0:s1] = seg
        self._line_cursor += n_lines
        return out

    def _line_waveform(
        self,
        tl: np.ndarray,
        t_abs: np.ndarray,
        kind: str,
        row: int,
        lg: int,
    ) -> np.ndarray:
        std = self.std
        if kind in {"pre_eq", "post_eq"}:
            return equalizing_pulse(tl, std)
        if kind == "serr":
            return serration_pulse(tl, std)
        # 普通行 (blank / active): 行同步 + 消隐
        wave = sync_pulse(tl, std)
        theta = 2.0 * np.pi * std.fsc * t_abs
        sin_c = np.sin(theta)
        cos_c = np.cos(theta)
        # PAL V 分量与突发逐行倒相
        pal_sign = 1.0
        if std.pal:
            pal_sign = 1.0 if (lg % 2 == 0) else -1.0
        # 色同步突发, 位于后肩
        b0 = std.burst_start_us * _US
        b1 = b0 + std.burst_cycles / std.fsc
        burst_mask = (tl >= b0) & (tl < b1)
        if std.pal:
            burst = BURST_AMP * 0.7071 * (pal_sign * sin_c - cos_c)
        else:
            burst = -BURST_AMP * sin_c
        wave = wave + np.where(burst_mask, burst, 0.0)
        if kind == "active":
            # 真隔行: 场0=源图偶数行, 场1=奇数行 (row 为场内行号)
            field = int((lg + 0.5) // std.field_lines) % 2
            src_row = 2 * row + field
            if 0 <= src_row < self.height:
                a0 = std.active_start_us * _US
                a1 = a0 + std.active_us * _US
                am = (tl >= a0) & (tl < a1)
                if np.any(am):
                    # 将该行像素插值到有效窗口的采样网格上
                    frac = (tl[am] - a0) / (std.active_us * _US)
                    px = np.clip(frac * self.width, 0, self.width - 1)
                    yv: NDArray[np.float64] = np.interp(
                        px, np.arange(self.width), self.y[src_row]
                    )
                    uv: NDArray[np.float64] = np.interp(
                        px, np.arange(self.width), self.u[src_row]
                    )
                    vv: NDArray[np.float64] = np.interp(
                        px, np.arange(self.width), self.v[src_row]
                    )
                    chroma: NDArray[np.float64] = (
                        uv * sin_c[am] + pal_sign * vv * cos_c[am]
                    )
                    wave[am] = yv + chroma
        return wave


def _fit_image(*, image: Image.Image, width: int, height: int) -> Image.Image:
    """把任意图像缩放/填充到有效分辨率 (4:3 显示空间保比, 黑边填充)."""
    img = image.convert("RGB")
    iw, ih = img.size
    aspect = iw / ih
    disp_w, disp_h = 4.0, 3.0
    if aspect >= disp_w / disp_h:
        dw, dh = disp_w, disp_w / aspect
    else:
        dh, dw = disp_h, disp_h * aspect
    nw = max(1, round(dw / disp_w * width))
    nh = max(1, round(dh / disp_h * height))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(img, ((width - nw) // 2, (height - nh) // 2))
    return canvas
