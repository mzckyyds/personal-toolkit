"""将解码后的场渲染为 CRT 屏幕画面."""

# numpy 数组运算的类型存根返回 Any 形状, 以下两项 unknown 检查对纯数值数组代码不适用
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .receiver import FieldDecode
    from .standards import Standard

__all__ = [
    "CRTDisplay",
]

_SHIFT_EPS = 0.5  # 小于半像素的水平偏移忽略
_VSHIFT_EPS = 0.01  # 小于该行数的垂直滚动忽略
_GRAIN_EPS = 1e-4  # 颗粒强度下限
_BLOOM_TH = 0.55  # 辉光亮度阈值
_BLOOM_DIVISOR = 220.0  # 辉光模糊半径相对屏幕尺寸


def _yuv_to_rgb(
    y: NDArray[np.float64], u: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    """YUV -> RGB (BT.601 系数), 裁剪到 0..1."""
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _shift_rows(
    img: NDArray[np.float64], offsets: NDArray[np.float64]
) -> NDArray[np.float64]:
    """逐行水平循环偏移 (像素), 实现同步失锁的斜向撕裂."""
    n, w = img.shape[:2]
    if not np.any(np.abs(offsets) > _SHIFT_EPS):
        return img
    out: NDArray[np.float64] = np.empty_like(img)
    x: NDArray[np.intp] = np.arange(w)
    for i in range(n):
        off = float(offsets[i])
        if abs(off) < _SHIFT_EPS:
            out[i] = img[i]
            continue
        # 线性插值的循环移位
        src: NDArray[np.float64] = (x - off) % w
        i0: NDArray[np.intp] = np.floor(src).astype(np.intp)
        frac: NDArray[np.float64] = src - i0
        i1: NDArray[np.intp] = (i0 + 1) % w
        out[i] = img[i, i0] * (1 - frac)[..., None] + img[i, i1] * frac[..., None]
    return out


@dataclass
class CRTDisplay:
    """荧光粉屏模型: 余辉, 扫描线, 辉光, 曲面几何与暗角."""

    std: Standard
    width: int
    height: int
    decay: float = 0.55  # 荧光粉每场衰减 (越小余辉越长)
    scanline_depth: float = 0.28
    bloom: float = 0.35
    curvature: float = 0.06
    vignette: float = 0.32
    grain: float = 0.02
    seed: int = 0
    phosphor: NDArray[np.float64] = field(init=False)
    _frame: NDArray[np.float64] = field(init=False)
    _geom_y: NDArray[np.float64] = field(init=False)
    _geom_x: NDArray[np.float64] = field(init=False)
    _vig: NDArray[np.float64] = field(init=False)
    _scan: NDArray[np.float64] = field(init=False)
    _last_noise: float = 0.0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(3))

    def __post_init__(self) -> None:
        """初始化荧光粉缓冲与几何/扫描线查找表."""
        self._rng = np.random.default_rng(self.seed + 3)
        self.phosphor = np.zeros((self.height, self.width, 3), dtype=np.float64)
        self._frame = np.zeros(
            (self.std.active_lines, self.std.active_width, 3), dtype=np.float64
        )
        self._build_geometry()
        self._build_scanlines()

    def _build_geometry(self) -> None:
        h, w = self.height, self.width
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        nx = (xx / w - 0.5) * 2.0
        ny = (yy / h - 0.5) * 2.0
        r2 = nx * nx + ny * ny
        k = self.curvature
        # 桶形畸变, 采样坐标向外扩
        sx = nx * (1.0 + k * r2)
        sy = ny * (1.0 + k * r2)
        self._geom_x = (sx / 2.0 + 0.5) * w
        self._geom_y = (sy / 2.0 + 0.5) * h
        # 暗角与圆角遮罩
        vig = 1.0 - self.vignette * np.clip(r2 - 0.35, 0.0, 1.0) / 0.65
        corner = np.clip((1.0 - np.abs(nx)) * (1.0 - np.abs(ny)) * 6.0, 0.0, 1.0)
        self._vig = (vig * corner)[..., None]

    def _build_scanlines(self) -> None:
        h = self.height
        n_lines = self.std.active_lines
        period = h / n_lines
        y = np.arange(h)
        # 扫描线, 行间略暗
        phase = 2.0 * np.pi * (y % period) / period
        scan = 1.0 - self.scanline_depth * (0.5 - 0.5 * np.cos(phase))
        self._scan = scan[:, None, None]

    def paint_field(self, *, fd: FieldDecode, field_index: int) -> None:
        """把一场解码内容写入隔行帧缓冲; 第二场到达时刷新荧光粉."""
        self._last_noise = fd.noise_sigma
        rgb = _yuv_to_rgb(fd.luma, fd.u, fd.v)
        rgb = _shift_rows(rgb, fd.h_offsets)
        # 垂直滚动, 环绕取样
        nfield = rgb.shape[0]
        vshift = fd.v_offset
        if abs(vshift) > _VSHIFT_EPS:
            src: NDArray[np.float64] = (np.arange(nfield) - vshift) % nfield
            i0: NDArray[np.intp] = np.floor(src).astype(np.intp)
            frac: NDArray[np.float64] = (src - i0)[:, None, None]
            i1: NDArray[np.intp] = (i0 + 1) % nfield
            rgb = rgb[i0] * (1 - frac) + rgb[i1] * frac
        # 隔行合成: 场0=偶数行, 场1=奇数行
        self._frame[field_index::2] = rgb
        if field_index == 1:
            # 放大到输出分辨率, 双线性保持 CRT 柔和
            img = self._resize(self._frame, self.height, self.width)
            # 荧光粉衰减与激励, 峰值保持产生余辉
            self.phosphor *= self.decay
            np.maximum(self.phosphor, img, out=self.phosphor)

    @staticmethod
    def _resize(img: NDArray[np.float64], h: int, w: int) -> NDArray[np.float64]:
        ih, iw = img.shape[:2]
        ys: NDArray[np.float64] = np.linspace(0, ih - 1, h)
        xs: NDArray[np.float64] = np.linspace(0, iw - 1, w)
        y0: NDArray[np.intp] = np.floor(ys).astype(np.intp)
        x0: NDArray[np.intp] = np.floor(xs).astype(np.intp)
        fy: NDArray[np.float64] = (ys - y0)[:, None, None]
        fx: NDArray[np.float64] = (xs - x0)[None, :, None]
        y1: NDArray[np.intp] = np.minimum(y0 + 1, ih - 1)
        x1: NDArray[np.intp] = np.minimum(x0 + 1, iw - 1)
        a = img[y0][:, x0]
        b = img[y0][:, x1]
        c = img[y1][:, x0]
        d = img[y1][:, x1]
        out: NDArray[np.float64] = (
            a * (1 - fy) * (1 - fx)
            + b * (1 - fy) * fx
            + c * fy * (1 - fx)
            + d * fy * fx
        )
        return out

    def snapshot(self) -> NDArray[np.uint8]:
        """读出当前屏幕画面, 叠加辉光/扫描线/曲面/暗角/颗粒, 返回 uint8 RGB."""
        img = self.phosphor.copy()
        # 辉光, 亮区扩散
        if self.bloom > 0:
            bright = np.clip(img - _BLOOM_TH, 0.0, 1.0)
            glow = gaussian_filter(
                bright,
                sigma=(self.height / _BLOOM_DIVISOR, self.width / _BLOOM_DIVISOR, 0),
            )
            img = img + self.bloom * glow
        # 扫描线
        img = img * self._scan
        # 曲面几何
        img = self._warp(img)
        # 暗角
        img = img * self._vig
        # 细颗粒, 显示噪声随信号噪声增强
        g = self.grain * (0.4 + min(self._last_noise, 1.0))
        if g > _GRAIN_EPS:
            img = img + self._rng.normal(0.0, g, img.shape)
        # 轻微整体柔化, CRT 非像素锐利
        img = gaussian_filter(img, sigma=(0.5, 0.5, 0))
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _warp(self, img: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = np.empty_like(img)
        for c in range(3):
            out[..., c] = map_coordinates(
                img[..., c],
                [self._geom_y, self._geom_x],
                order=1,
                mode="constant",
                cval=0.0,
            )
        return out
