"""控件空间恢复时间线: 把"无信号 -> 清晰"建模为一组物理旋钮从随机失配收敛到标称.

真实老电视的恢复是人去调几个旋钮: 天线(信号强度/重影)、行频 H.HOLD、场频 V.HOLD、
微调 fine-tune. 每次运行随机生成一组"失配初值"和一组"类人收敛轨迹"(分阶段、随机
顺序、过冲回拧), 让旋钮值逐步接近标称工作点. 画面不插值, 由接收机对瞬时旋钮值的
非线性响应涌现 (同步捕获/消色阈值), 因此每次效果不同但物理过程一致.

旋钮 -> 仿真量:
  antenna  -> 信道 cnr_db / ghost / fade (0=无信号, 1=干净)
  h_hold   -> 行自由振荡频偏 (撕裂)
  v_hold   -> 场自由振荡频偏 (滚动)
  tune     -> 本振/副载波频偏 Hz (彩色失锁/旋涡)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channel import ChannelParams

__all__ = [
    "ControlState",
    "RecoveryTimeline",
]

_DROPOUT_PROB = 0.5  # 弱信号阶段候选点掉信号概率
_CNR_MIN = -4.0  # antenna=0 对应载噪比
_CNR_SPAN = 47.0  # antenna=1 对应载噪比增量 (-4 -> 43)
_GHOST_MAX = 0.5  # antenna=0 时主重影幅度
_FADE_MAX = 0.5  # antenna=0 时衰落深度
_INTERF_MAX = 0.05  # antenna=0 时干扰幅度
_H_HOLD_MAX = 0.09  # 行频失配初值上限 (相对行频)
_V_HOLD_MAX = 0.10  # 场频失配初值上限 (相对场频)
_TUNE_MAX = 1.5e6  # 微调失配初值上限 (Hz)
_TUNE_MIN = 0.3e6  # 微调失配初值下限 (Hz)
_H_FIRST_PROB = 0.5  # 行频先于场频收敛的概率


@dataclass
class ControlState:
    """某一时刻的旋钮值 (接收机/信道据此响应)."""

    antenna: float  # 0..1 信号质量
    h_hold: float  # 行频偏 (相对行频,  signed)
    v_hold: float  # 场频偏 (相对场频,  signed)
    tune: float  # 本振频偏 (Hz, signed)


@dataclass
class _Knob:
    """单个旋钮的初值/目标/收敛窗口/类人轨迹参数."""

    start: float
    end: float
    initial: float
    target: float
    osc: float
    cycles: float
    jitter: float
    phase: float

    def value_at(self, *, frac: float, t: float) -> float:
        """返回 frac 时刻的旋钮值: 失配初值经类人轨迹收敛到目标."""
        if frac <= self.start:
            u = 0.0
        elif frac >= self.end:
            u = 1.0
        else:
            u = (frac - self.start) / max(self.end - self.start, 1e-6)
        if u >= 1.0:
            return self.target
        base = (1.0 - u) ** 1.4
        # 过冲回拧与高频犹豫抖动, 模拟人手调节
        r = base * (1.0 + self.osc * np.sin(2 * np.pi * self.cycles * u))
        r += self.jitter * (1.0 - u) * np.sin(2 * np.pi * 7.3 * t + self.phase)
        return self.target + (self.initial - self.target) * r


@dataclass
class RecoveryTimeline:
    """随机失配初值 + 分阶段类人收敛, 输出逐时刻旋钮值与信道参数."""

    duration: float  # 恢复过程跨度 (秒); 图片=总时长, 视频=片头
    seed: int = 0
    snow_until: float = 0.16
    lock_by: float = 0.62
    _rng: np.random.Generator = field(init=False)
    _antenna: _Knob = field(init=False)
    _h_hold: _Knob = field(init=False)
    _v_hold: _Knob = field(init=False)
    _tune: _Knob = field(init=False)
    _dropouts: list[tuple[float, float, float]] = field(init=False)

    def __post_init__(self) -> None:
        """随机生成初值/顺序/轨迹, 全部由 seed 决定 (可复现)."""
        rng = np.random.default_rng(self.seed)
        self._rng = rng
        s = self.snow_until
        span = max(self.lock_by - s, 1e-6)

        def win(lo: float, hi: float) -> tuple[float, float]:
            a = s + span * float(rng.uniform(lo, hi))
            b = a + span * float(rng.uniform(0.25, 0.45))
            return a, min(b, self.lock_by)

        # 天线最先 (信号闯入), hold 次之 (随机先后), 微调最后 (彩色绽开)
        self._antenna = _Knob(
            start=s,
            end=s + span * float(rng.uniform(0.4, 0.6)),
            initial=float(rng.uniform(0.0, 0.06)),
            target=1.0,
            osc=float(rng.uniform(0.05, 0.2)),
            cycles=float(rng.uniform(0.5, 1.5)),
            jitter=float(rng.uniform(0.0, 0.03)),
            phase=float(rng.uniform(0, 6.28)),
        )
        h_first = bool(rng.random() < _H_FIRST_PROB)
        h_lo, h_hi = (0.10, 0.30) if h_first else (0.25, 0.45)
        v_lo, v_hi = (0.25, 0.45) if h_first else (0.10, 0.30)
        hs, he = win(h_lo, h_hi)
        vs, ve = win(v_lo, v_hi)
        self._h_hold = _Knob(
            start=hs,
            end=he,
            initial=float(rng.uniform(0.03, _H_HOLD_MAX)) * rng.choice([-1.0, 1.0]),
            target=0.0,
            osc=float(rng.uniform(0.1, 0.35)),
            cycles=float(rng.uniform(1.0, 2.5)),
            jitter=float(rng.uniform(0.0, 0.04)),
            phase=float(rng.uniform(0, 6.28)),
        )
        self._v_hold = _Knob(
            start=vs,
            end=ve,
            initial=float(rng.uniform(0.03, _V_HOLD_MAX)) * rng.choice([-1.0, 1.0]),
            target=0.0,
            osc=float(rng.uniform(0.1, 0.35)),
            cycles=float(rng.uniform(1.0, 2.5)),
            jitter=float(rng.uniform(0.0, 0.04)),
            phase=float(rng.uniform(0, 6.28)),
        )
        ts, te = win(0.45, 0.65)
        self._tune = _Knob(
            start=ts,
            end=te,
            initial=float(rng.uniform(_TUNE_MIN, _TUNE_MAX)) * rng.choice([-1.0, 1.0]),
            target=0.0,
            osc=float(rng.uniform(0.1, 0.3)),
            cycles=float(rng.uniform(1.0, 2.0)),
            jitter=float(rng.uniform(0.0, 0.03)),
            phase=float(rng.uniform(0, 6.28)),
        )
        # 弱信号阶段偶发掉信号
        self._dropouts = []
        t = s + 0.06
        while t < self.lock_by - 0.05:
            if rng.random() < _DROPOUT_PROB:
                self._dropouts.append(
                    (
                        t,
                        float(rng.uniform(0.02, 0.06)),
                        float(rng.uniform(6.0, 14.0)),
                    )
                )
            t += float(rng.uniform(0.06, 0.14))

    def controls_at(self, *, t: float) -> ControlState:
        """返回时刻 t 的旋钮值."""
        frac = float(np.clip(t / self.duration, 0.0, 1.0))
        return ControlState(
            antenna=float(np.clip(self._antenna.value_at(frac=frac, t=t), 0.0, 1.0)),
            h_hold=self._h_hold.value_at(frac=frac, t=t),
            v_hold=self._v_hold.value_at(frac=frac, t=t),
            tune=self._tune.value_at(frac=frac, t=t),
        )

    def params_at(self, *, t: float) -> ChannelParams:
        """由 antenna 旋钮派生信道参数, 叠加呼吸抖动与偶发 dropout."""
        frac = float(np.clip(t / self.duration, 0.0, 1.0))
        a = self.controls_at(t=t).antenna
        cnr = _CNR_MIN + a * _CNR_SPAN
        ghost = (1.0 - a) * _GHOST_MAX
        fade = (1.0 - a) * _FADE_MAX
        interf = (1.0 - a) * _INTERF_MAX
        weak = np.clip((22.0 - cnr) / 22.0, 0.0, 1.0)
        breath = weak * a  # 无载波时不呼吸
        cnr += breath * 2.2 * np.sin(2 * np.pi * 0.9 * t + 1.3)
        cnr += breath * 1.1 * np.sin(2 * np.pi * 2.3 * t + 0.7)
        for t0, dur, depth in self._dropouts:
            if t0 <= frac <= t0 + dur:
                u = (frac - t0) / dur
                cnr -= depth * np.sin(np.pi * u)
                ghost += 0.2 * np.sin(np.pi * u)
        cnr = float(np.clip(cnr, -6.0, 50.0))
        return ChannelParams(
            cnr_db=cnr,
            ghost_amp=float(np.clip(ghost, 0.0, 0.6)),
            ghost_delay_us=1.6,
            ghost2_amp=float(np.clip(ghost * 0.4, 0.0, 0.3)),
            ghost2_delay_us=4.2,
            fade_depth=float(np.clip(fade, 0.0, 0.7)),
            fade_rate_hz=0.7,
            interference_amp=float(np.clip(interf, 0.0, 0.1)),
            interference_beat_hz=15_734.0 * 0.97,
        )
