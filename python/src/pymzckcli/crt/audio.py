"""CRT 音频: 无信号雪花嘶声随信号锁定逐渐淡出, 叠加轻微行频啸叫.

真实老式电视在无信号时扬声器发出响亮的白噪声嘶声; 信号锁定后嘶声消失.
这里用逐输出帧记录的噪声电平驱动嘶声包络, 并叠加 ~15.7kHz 行输出变压器啸叫.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

__all__ = [
    "generate_hiss",
]

_SAMPLE_RATE = 44100
_NOISE_REF = 1.3  # 雪花时接收机噪声标准差参考值
_HISS_GAIN = 0.55  # 嘶声最大响度
_NOISE_FLOOR = 0.006  # 锁定后保留的轻微本底
_WHINE_FREQ = 15734.0  # NTSC 行频, 行输出变压器啸叫
_WHINE_GAIN = 0.006
_HIGHPASS_TAPS = 64  # 滑动平均窗长, 用于去除低频隆隆声
_PEAK_LIMIT = 0.95  # 归一化峰值上限, 防止削波
_INPUT_GAIN = 0.9  # 视频原声响度 (锁定后)
_SPEAKER_TAPS = 10  # 老式扬声器低通窗长 (~4kHz)


def generate_hiss(  # noqa: PLR0913
    *,
    levels: list[tuple[float, float]],
    duration: float,
    path: Path,
    seed: int = 0,
    whine: bool = True,
    input_audio: Path | None = None,
) -> Path:
    """根据 (时间, 噪声电平) 序列生成嘶声音频 WAV, 返回路径.

    levels 中噪声电平取自接收机 AGC 后的噪声标准差 (雪花时 ~1.3, 锁定时 ~0.04).
    """
    rng = np.random.default_rng(seed)
    n = round(duration * _SAMPLE_RATE)
    t = np.arange(n) / _SAMPLE_RATE

    if levels:
        ts = np.array([lv[0] for lv in levels])
        vs = np.array([lv[1] for lv in levels])
        env = np.interp(t, ts, vs)
    else:
        env = np.zeros(n)
    # 噪声电平 -> 嘶声权重 (雪花=1, 锁定=0); 留少量本底
    w = np.clip(env / _NOISE_REF, 0.0, 1.0)
    amp = w * _HISS_GAIN + _NOISE_FLOOR

    # 白噪声嘶声, 轻微高通去除低频隆隆声, 更接近电视扬声器质感
    noise = rng.normal(0.0, 1.0, n)
    kernel = np.ones(_HIGHPASS_TAPS) / _HIGHPASS_TAPS
    noise = noise - np.convolve(noise, kernel, mode="same")
    hiss = noise * amp

    signal = hiss
    if input_audio is not None:
        # 视频输入: 锁定后淡入原声 (老式扬声器质感), 与嘶声交叉淡化
        orig = _load_input_audio(path=input_audio, n=n)
        signal = signal + orig * (1.0 - w) * _INPUT_GAIN
    if whine:
        # 行输出变压器啸叫, 极低音量, 仅无信号时明显
        signal = signal + _WHINE_GAIN * w * np.sin(2.0 * np.pi * _WHINE_FREQ * t)

    peak = float(np.max(np.abs(signal)))
    if peak > _PEAK_LIMIT:
        signal = signal * (_PEAK_LIMIT / peak)
    pcm = (signal * 32767.0).astype(np.int16)
    wavfile.write(str(path), _SAMPLE_RATE, pcm)
    return path


def _load_input_audio(*, path: Path, n: int) -> np.ndarray:
    """解码视频原声为单声道浮点并低通 (老式扬声器), 无音轨时返回静音."""
    try:
        with tempfile.TemporaryDirectory(prefix="crt_in_") as tmp:
            wav = Path(tmp) / "in.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(_SAMPLE_RATE),
                str(wav),
            ]
            result = subprocess.run(cmd, check=False, stderr=subprocess.DEVNULL)  # noqa: S603
            if result.returncode != 0 or not wav.exists():
                return np.zeros(n)
            _rate, data = wavfile.read(str(wav))
    except (OSError, ValueError):
        return np.zeros(n)
    x = np.asarray(data, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        x = x / 32768.0
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    x = x[:n]
    kernel = np.ones(_SPEAKER_TAPS) / _SPEAKER_TAPS
    return np.convolve(x, kernel, mode="same")
