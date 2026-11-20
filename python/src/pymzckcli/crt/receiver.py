"""接收机: 从受损的复合信号中恢复图像, 并仿真同步/AGC/色度电路的动态.

真实感的核心: 撕裂, 滚动, 雪花, 彩色消色不是叠加的特效, 而是下列电路在弱信号下
的自然行为:

* AGC (自动增益控制): 以同步头为键控. 信号消失时增益缓慢爬升到最大, 噪声被放大
  成满屏雪花; 信号出现时增益快速回落.
* 同步分离 + 行 AFC (飞轮锁相): 干净同步则锁定; 噪声淹没同步则行振荡器自由运行,
  每行水平偏移累积, 形成斜向撕裂/拉扯.
* 场同步积分器 + 场振荡器: 检测不到场同步则场频自由运行, 画面上下滚动.
* 色度解调 + 消色电路 (color killer): 色同步突发弱则关闭彩色 (黑白), 突发相位抖动
  则色调漂移.

视频内容在标称行网格上解调 (信号里客观存在), 而"画在哪里"由恢复/自由运行的偏转
振荡器决定: 锁定则稳定, 失锁则撕裂/滚动.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .standards import line_kind

if TYPE_CHECKING:
    from .standards import Standard

__all__ = [
    "FieldDecode",
    "Receiver",
]

_US = 1e-6  # 微秒 -> 秒
_EPS = 1e-6
_MIN_SYNC_WIN = 4  # 同步窗口最少样本数
_MIN_BLANK_WIN = 2  # 消隐窗口最少样本数
_SYNC_TARGET = 0.40  # AGC 归一化后的同步头深度
_SYNC_COHER_TH = 0.7  # 判定同步存在的相干性阈值
_MIN_SYNC_DEPTH = 0.02  # 判定同步存在的最小深度
_AGC_ATTACK = 0.5  # 有同步时增益收敛速度 (快)
_AGC_DECAY = 0.06  # 无同步时增益爬升速度 (慢)
_NOISE_WEIGHT = 5.0  # 同步质量中噪声的权重
_H_LOCK_ON = 0.62  # 行锁定上限阈值
_H_LOCK_OFF = 0.42  # 行锁定下限阈值
_V_LOCK_ON = 0.52  # 场锁定上限阈值
_V_LOCK_OFF = 0.34  # 场锁定下限阈值
_H_LOCK_SMOOTH = 0.35  # 行锁定平滑系数
_V_LOCK_SMOOTH = 0.22  # 场锁定平滑系数
_BURST_NOISE_WEIGHT = 1.5  # 消色判定中噪声的权重
_BURST_SMOOTH = 0.3  # 突发电平平滑系数
_COLOR_OFF_TH = 0.30  # 消色关闭阈值 (迟滞下限)
_COLOR_ON_TH = 0.45  # 消色开启阈值 (迟滞上限)
_LUMA_BW = 2.2e6  # 亮度低通带宽 (Hz)
_CHROMA_BW = 1.1e6  # 色度解调低通带宽 (Hz)
_SNOW_DC_GATE = 0.5  # 低于此行锁定才叠加雪花亮度偏置
_SNOW_DC_GAIN = 0.42  # 雪花亮度偏置增益
_KICK_GATE = 0.4  # 低于此行锁定才可能出现大相位踢
_BIG_KICK_PROB = 0.02  # 大相位踢概率 (噪声提前触发行振荡器)
_KICK_SIGMA = 0.12  # 常规相位踢标准差
_BIG_KICK_SIGMA = 0.3  # 大相位踢标准差
_PULLBACK_GATE = 0.8  # 高于此锁定值时相位缓慢拉回
_H_PULLBACK = 0.85  # 行相位拉回系数
_V_PULLBACK = 0.8  # 场相位拉回系数
_DEFAULT_NOISE = 0.3  # 无法估计噪声时的兜底值
_H_CAP = 0.02  # 行 AFC 捕获范围 (相对行频), 超出则无法锁定
_V_CAP = 0.03  # 场 AFC 捕获范围 (相对场频)
_TUNE_BW = 0.6e6  # 微调失谐衰减带宽 (Hz)


@dataclass
class FieldDecode:
    """一场解码结果: 视频内容 + 偏转位置 + 电路状态."""

    luma: np.ndarray  # (n_lines, width) 0..1
    u: np.ndarray
    v: np.ndarray
    h_offsets: np.ndarray  # (n_lines,) 每行水平偏移 (像素, 浮点)
    v_offset: float  # 垂直滚动偏移 (行)
    h_lock: float  # 0..1 行锁定质量
    v_lock: float  # 0..1 场锁定质量
    color_on: bool
    noise_sigma: float  # 归一化噪声强度 (供显示颗粒)
    agc_gain: float


def _lp(sig: np.ndarray, fs: float, bw: float) -> np.ndarray:
    """高斯低通近似, 带宽 bw (Hz)."""
    sigma = max(fs / (2.0 * np.pi * bw), 0.5)
    return gaussian_filter1d(sig, sigma, mode="nearest")


@dataclass
class Receiver:
    """仿真接收机的 AGC/同步/色度/偏转电路, 逐帧输出两场解码结果."""

    std: Standard
    seed: int = 0
    # AGC
    gain: float = 12.0
    gain_max: float = 26.0
    gain_min: float = 0.6
    # 行/场"保持"电位器偏置 (自由运行频率相对标称的偏差)
    h_hold_error: float = 0.035
    v_hold_error: float = 0.045
    tune_offset: float = 0.0  # 本振/副载波频偏 (Hz), 微调旋钮
    # 锁定状态, 经平滑
    h_lock: float = 0.0
    v_lock: float = 0.0
    color_on: bool = False
    noise_sigma: float = 0.0
    _burst_level: float = 0.0
    # 偏转相位误差累积
    _h_phase_err: float = 0.0
    _v_phase_err: float = 0.0
    _sample_cursor: int = 0
    _frame_count: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    def __post_init__(self) -> None:
        """按 seed 初始化随机源 (可复现)."""
        self._rng = np.random.default_rng(self.seed + 7)

    def process_frame(self, *, composite: np.ndarray) -> list[FieldDecode]:
        """处理一帧复合信号, 返回两场的解码与电路状态."""
        std = self.std
        x = np.asarray(composite, dtype=np.float64)
        spl = std.samples_per_line
        n_lines = std.total_lines

        # AGC, 用原始信号估计同步头深度
        depth_raw, sync_frac = self._estimate_sync_depth(x, spl, n_lines)
        self._update_agc(depth_raw, sync_frac)
        g = self.gain
        sig = x * g

        # 噪声与同步质量, 推导锁定状态
        noise_sigma = self._estimate_noise(sig, spl, n_lines)
        self.noise_sigma = noise_sigma
        quality = self._sync_quality(sig, spl, n_lines, noise_sigma)
        self._update_locks(quality=quality)

        # 色同步突发幅度, 推导消色
        burst = self._estimate_burst(sig, spl, n_lines)
        tune_factor = float(np.exp(-((self.tune_offset / _TUNE_BW) ** 2)))
        self._update_color(burst * tune_factor, noise_sigma)

        # 逐行解调内容 + 偏转偏移
        width = std.active_width
        nfield = std.active_lines_per_field
        fields: list[FieldDecode] = []
        line_starts = np.round(np.arange(n_lines + 1) * spl).astype(np.int64)
        base_idx = self._sample_cursor
        for f in range(2):
            luma = np.zeros((nfield, width))
            uu = np.zeros((nfield, width))
            vv = np.zeros((nfield, width))
            h_off = np.zeros(nfield)
            row_i = 0
            for k in range(n_lines):
                lg = self._frame_count * n_lines + k
                kind, row = line_kind(std, lg)
                if kind != "active" or not (0 <= row < nfield):
                    continue
                # 判断该行属于哪一场
                fld = int((lg + 0.5) // std.field_lines) % 2
                if fld != f:
                    continue
                s0 = line_starts[k]
                s1 = line_starts[k + 1]
                seg = sig[s0:s1]
                y, u, v = self._decode_line(seg, base_idx + s0, lg, noise_sigma)
                if row_i < nfield:
                    luma[row_i] = y
                    uu[row_i] = u
                    vv[row_i] = v
                    h_off[row_i] = self._h_offset_for_line(noise_sigma)
                    row_i += 1
            v_off = self._v_phase_err * nfield
            fields.append(
                FieldDecode(
                    luma=luma,
                    u=uu,
                    v=vv,
                    h_offsets=h_off,
                    v_offset=v_off,
                    h_lock=self.h_lock,
                    v_lock=self.v_lock,
                    color_on=self.color_on,
                    noise_sigma=noise_sigma,
                    agc_gain=g,
                )
            )
        # 推进场相位, 每帧两场
        self._advance_vertical()
        self._sample_cursor += round(n_lines * spl)
        self._frame_count += 1
        return fields

    def _estimate_sync_depth(
        self, x: np.ndarray, spl: float, n_lines: int
    ) -> tuple[float, float]:
        """估计同步头深度与同步存在置信度.

        真实同步是整段低电平的相干脉冲, 噪声只是随机起伏. 用同步窗口均值相对消隐
        的下沉量除以其标准差 (相干性) 来区分二者, 避免 AGC 把噪声误判为同步.
        """
        std = self.std
        sync_n = round(std.sync_us * _US * std.fs)
        depths: list[float] = []
        coher: list[float] = []
        for k in range(0, n_lines, 3):
            s0 = round(k * spl)
            win = x[s0 : s0 + sync_n]
            if win.size < _MIN_SYNC_WIN:
                continue
            b0 = s0 + sync_n + round(0.3 * _US * std.fs)
            b1 = s0 + round(std.burst_start_us * _US * std.fs)
            blank = x[b0:b1]
            if blank.size < _MIN_BLANK_WIN:
                continue
            bl = float(np.median(blank))
            tip = float(np.mean(np.sort(win)[: max(2, win.size // 4)]))
            depths.append(bl - tip)
            # 相干性, 同步窗口均值低于消隐的程度除以其起伏
            wmean = float(np.mean(win))
            wstd = float(np.std(win)) + _EPS
            coher.append((bl - wmean) / wstd)
        if not depths:
            return 0.0, 0.0
        return float(np.median(depths)), float(np.median(coher))

    def _update_agc(self, depth_raw: float, sync_frac: float) -> None:
        # 有相干同步则快速把同步头归一化, 无同步则缓慢爬向最大增益
        if sync_frac > _SYNC_COHER_TH and depth_raw > _MIN_SYNC_DEPTH:
            target = float(
                np.clip(_SYNC_TARGET / depth_raw, self.gain_min, self.gain_max)
            )
            self.gain += (target - self.gain) * _AGC_ATTACK
        else:
            self.gain += (self.gain_max - self.gain) * _AGC_DECAY
        self.gain = float(np.clip(self.gain, self.gain_min, self.gain_max))

    def _estimate_noise(self, sig: np.ndarray, spl: float, n_lines: int) -> float:
        """用行同步窗口起伏估计噪声 sigma (同步头平坦, 其标准差即噪声)."""
        std = self.std
        sync_n = round(std.sync_us * _US * std.fs)
        vals: list[float] = []
        for k in range(0, n_lines, 5):
            s0 = round(k * spl)
            seg = sig[s0 : s0 + sync_n]
            if seg.size > _MIN_SYNC_WIN:
                vals.append(float(np.std(seg)))
        if not vals:
            return _DEFAULT_NOISE
        return float(np.median(vals))

    def _sync_quality(
        self, sig: np.ndarray, spl: float, n_lines: int, noise: float
    ) -> float:
        """同步头幅度相对噪声, 归一化为 0..1 质量."""
        std = self.std
        sync_n = round(std.sync_us * _US * std.fs)
        tips: list[float] = []
        for k in range(0, n_lines, 3):
            s0 = round(k * spl)
            win = sig[s0 : s0 + sync_n]
            if win.size < _MIN_SYNC_WIN:
                continue
            tips.append(float(np.mean(np.sort(win)[: max(2, win.size // 4)])))
        if not tips:
            return 0.0
        tip = float(np.median(tips))
        depth = max(0.0, -tip)  # 同步头应为负
        q = depth / (depth + _NOISE_WEIGHT * noise + _EPS)
        return float(np.clip(q, 0.0, 1.0))

    def _update_locks(self, *, quality: float) -> None:
        # 行锁定, 质量高则锁定, 带迟滞与抖动 (弱信号时锁定"猎振")
        h_span = _H_LOCK_ON - _H_LOCK_OFF
        if quality > _H_LOCK_ON:
            h_target = 1.0
        elif quality < _H_LOCK_OFF:
            h_target = 0.0
        else:
            h_target = float(self._rng.random() < (quality - _H_LOCK_OFF) / h_span)
        self.h_lock += (h_target - self.h_lock) * _H_LOCK_SMOOTH
        # 场锁定阈值略低, 积分器抗噪, 时间常数更慢
        v_span = _V_LOCK_ON - _V_LOCK_OFF
        if quality > _V_LOCK_ON:
            v_target = 1.0
        elif quality < _V_LOCK_OFF:
            v_target = 0.0
        else:
            v_target = float(self._rng.random() < (quality - _V_LOCK_OFF) / v_span)
        self.v_lock += (v_target - self.v_lock) * _V_LOCK_SMOOTH
        self.h_lock = float(np.clip(self.h_lock, 0, 1))
        self.v_lock = float(np.clip(self.v_lock, 0, 1))

    def _estimate_burst(self, sig: np.ndarray, spl: float, n_lines: int) -> float:
        """估计色同步突发幅度 (后肩窗口标准差)."""
        std = self.std
        b0 = round(std.burst_start_us * _US * std.fs)
        b1 = b0 + round(std.burst_cycles / std.fsc * std.fs)
        amps: list[float] = []
        for k in range(0, n_lines, 3):
            s0 = round(k * spl)
            seg = sig[s0 + b0 : s0 + b1]
            if seg.size > _MIN_SYNC_WIN:
                amps.append(float(np.std(seg)))
        if not amps:
            return 0.0
        return float(np.median(amps))

    def _update_color(self, burst: float, noise: float) -> None:
        # 消色电路, 突发幅度需显著高于噪声才开彩色, 带迟滞
        level = burst / (burst + _BURST_NOISE_WEIGHT * noise + _EPS)
        self._burst_level += (level - self._burst_level) * _BURST_SMOOTH
        if self.color_on:
            if self._burst_level < _COLOR_OFF_TH:
                self.color_on = False
        elif self._burst_level > _COLOR_ON_TH:
            self.color_on = True

    def _decode_line(
        self, seg: np.ndarray, global_idx: int, lg: int, noise: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        std = self.std
        width = std.active_width
        a0 = round(std.active_start_us * _US * std.fs)
        a1 = a0 + round(std.active_us * _US * std.fs)
        active = seg[a0:a1]
        if active.size < width:
            active = np.pad(active, (0, width - active.size))
        active = active[:width]
        fs = std.fs
        # 亮度, 低通去除色度 (CRT 柔和带宽)
        y = _lp(active, fs, _LUMA_BW)
        y = np.clip((y - std.black_level) / (1.0 - std.black_level), 0.0, 1.0)
        # 无信号雪花亮度, AGC 满增益放大噪声, CRT 亮度偏置使雪花呈中等亮度而非死黑.
        # 仅在噪声主导 (无相干信号) 时叠加, 信号浮现后自然消退.
        if self.h_lock < _SNOW_DC_GATE:
            dc = _SNOW_DC_GAIN * float(np.clip(noise - 0.25, 0.0, 1.2))
            y = np.clip(y + dc * (1.0 - self.h_lock), 0.0, 1.0)
        # 色度, 正交解调 (4fsc 每样本相位增量 π/2)
        idx = global_idx + a0 + np.arange(width)
        theta = (np.pi / 2.0) * idx + 2 * np.pi * self.tune_offset * (idx / fs)
        sin_c = np.sin(theta)
        cos_c = np.cos(theta)
        pal_sign = 1.0 if (not std.pal or lg % 2 == 0) else -1.0
        u_raw = _lp(active * sin_c * 2.0, fs, _CHROMA_BW)
        v_raw = _lp(active * cos_c * 2.0 * pal_sign, fs, _CHROMA_BW)
        tune_factor = float(np.exp(-((self.tune_offset / _TUNE_BW) ** 2)))
        u_raw = u_raw * tune_factor
        v_raw = v_raw * tune_factor
        if not self.color_on:
            u_raw = np.zeros_like(u_raw)
            v_raw = np.zeros_like(v_raw)
        return y, u_raw, v_raw

    def _h_offset_for_line(self, noise: float) -> float:
        # 行频, 锁定则标称, 失锁则自由运行 (hold 偏置)
        h_cap = float(np.clip(1.0 - abs(self.h_hold_error) / _H_CAP, 0.0, 1.0))
        eff = self.h_lock * h_cap
        drift = (1.0 - eff) * self.h_hold_error
        # 噪声误触发产生随机相位踢, 失锁时撕裂; 偶发大踢模拟噪声提前触发
        kick = (1.0 - eff) * noise * self._rng.normal(0.0, _KICK_SIGMA)
        if eff < _KICK_GATE and self._rng.random() < _BIG_KICK_PROB:
            kick += (1.0 - eff) * self._rng.normal(0.0, _BIG_KICK_SIGMA)
        self._h_phase_err += drift + kick
        # 锁定时缓慢拉回零相位
        if eff > _PULLBACK_GATE:
            self._h_phase_err *= _H_PULLBACK
        width = self.std.active_width
        return float((self._h_phase_err % 1.0) * width)

    def _advance_vertical(self) -> None:
        # 每帧两场, 场相位误差累积
        v_cap = float(np.clip(1.0 - abs(self.v_hold_error) / _V_CAP, 0.0, 1.0))
        eff = self.v_lock * v_cap
        for _ in range(2):
            drift = (1.0 - eff) * self.v_hold_error
            self._v_phase_err += drift
            if eff > _PULLBACK_GATE:
                self._v_phase_err *= _V_PULLBACK
        # 保持在 [-0.5, 0.5) 便于滚动环绕
        self._v_phase_err = (self._v_phase_err + 0.5) % 1.0 - 0.5
