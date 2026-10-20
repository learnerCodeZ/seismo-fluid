"""features/build.py — 滑窗构造与标准化。

迁移自 dachuang_cnn_lstm/src/features/build.py，逐行一致。

设计要点（防时序泄漏）：
- 标准化统计量只从训练段估计，应用到测试段；
- 滑窗窗口 window=60、步长 1（与参照文献一致），均由配置传入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_fit(train: np.ndarray) -> tuple[float, float]:
    mu = float(np.nanmean(train))
    sd = float(np.nanstd(train))
    sd = sd if sd > 1e-12 else 1.0
    return mu, sd


def zscore_apply(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / sd


def make_windows(arr: np.ndarray, window: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """单变量滑窗。arr 形状 [T] → X [T-window-horizon+1, window], y [N]。"""
    arr = np.asarray(arr, dtype=float)
    n = len(arr) - window - horizon + 1
    if n <= 0:
        raise ValueError(f"序列长度 {len(arr)} 不足以构造 window={window} 的滑窗")
    idx = np.arange(window) + np.arange(n)[:, None]
    X = arr[idx]
    y = arr[idx[:, -1] + horizon]
    return X, y


def make_windows_multi(arr2d: np.ndarray, window: int, target_col: int = 0,
                       horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """多变量滑窗。arr2d 形状 [T, n_vars] → X [N, window, n_vars], y [N]（target_col 下一时刻）。"""
    arr2d = np.asarray(arr2d, dtype=float)
    n = len(arr2d) - window - horizon + 1
    if n <= 0:
        raise ValueError(f"序列长度 {len(arr2d)} 不足以构造 window={window} 的滑窗")
    idx = np.arange(window) + np.arange(n)[:, None]
    X = arr2d[idx]                                  # [N, window, n_vars]
    y = arr2d[idx[:, -1] + horizon, target_col]
    return X, y


def remove_annual_cycle(dates, values: np.ndarray,
                        fit_mask: np.ndarray | None = None) -> np.ndarray:
    """去除年变周期：按"年积日"（day-of-year）气候平均拟合，仅在 fit_mask 段估计。

    dates: pandas DatetimeIndex/Series；fit_mask: 训练段布尔数组（None=全部）。
    """
    dt = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    doy = dt.dt.dayofyear.to_numpy()
    v = np.asarray(values, dtype=float)
    mask = np.ones(len(v), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    clim = np.full(367, np.nan)
    for d in range(1, 367):
        sel = (doy == d) & mask
        if sel.sum() >= 3:
            clim[d] = np.nanmean(v[sel])
    # 环日滑窗填补（±7 天），保证闰年/缺日处也有气候值
    s = pd.Series(clim).interpolate(limit_direction="both")
    clim = s.to_numpy()[doy]  # clim[1..366] 按年积日取值
    return v - clim


def rolling_stats(values: np.ndarray, window: int) -> np.ndarray:
    """滑动均值与标准差。返回 [T, 2]，前 window-1 行为 NaN。"""
    s = pd.Series(np.asarray(values, dtype=float))
    mean = s.rolling(window, min_periods=1).mean().to_numpy()
    std = s.rolling(window, min_periods=1).std().fillna(0).to_numpy()
    return np.column_stack([mean, std])


def diff_features(values: np.ndarray) -> np.ndarray:
    """一阶差分（日变化）与二阶差分（加速度）。返回 [T, 2]，首行/前两行补 0。"""
    v = np.asarray(values, dtype=float)
    d1 = np.diff(v, prepend=v[0])
    d2 = np.diff(d1, prepend=d1[0])
    return np.column_stack([d1, d2])


def build_feature_matrix(values: np.ndarray, *, use_roll: bool = True,
                         roll_windows: tuple[int, ...] = (7, 14, 30),
                         use_diff: bool = True) -> tuple[np.ndarray, list[str]]:
    """把单变量序列扩成多通道特征矩阵。

    列顺序：raw, [roll_mean_w, roll_std_w]*, [d1, d2]
    特征只依赖过去值（rolling/diff 均向后看），与滑窗目标对齐后无未来泄漏。
    返回 (X [T, n_feat], names)。
    """
    v = np.asarray(values, dtype=float)
    cols = [v]
    names = ["raw"]
    if use_roll:
        for w in roll_windows:
            rs = rolling_stats(v, w)
            cols.extend([rs[:, 0], rs[:, 1]])
            names.extend([f"roll_mean_{w}", f"roll_std_{w}"])
    if use_diff:
        df = diff_features(v)
        cols.extend([df[:, 0], df[:, 1]])
        names.extend(["diff1", "diff2"])
    return np.column_stack(cols), names


def robust_fit(train: np.ndarray) -> tuple[float, float]:
    """RobustScaler：中位数 + IQR。对离群值比 z-score 稳。"""
    v = np.asarray(train, dtype=float)
    v = v[~np.isnan(v)]
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr if iqr > 1e-12 else float(np.std(v) or 1.0)
    scale = scale if scale > 1e-12 else 1.0
    return med, scale


def robust_apply(x: np.ndarray, med: float, scale: float) -> np.ndarray:
    return (np.asarray(x, dtype=float) - med) / scale
