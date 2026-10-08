"""ScalePainter phase ancestors. Derived from the verified research implementation."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from scalepainter.phase_geometry import _adaptive_bins, _rect_sum

def _unlock_rank(edge: int | None, formation_edges: Sequence[int]) -> int:
    """把首次解锁尺度转成可比较秩；没有解锁时位于所有形成尺度之后。"""
    if edge is None:
        return len(tuple(formation_edges))
    return tuple((int(value) for value in formation_edges)).index(int(edge))

def _ancestor_counts_from_integral(integral: np.ndarray, *, dx: int=0, dy: int=0) -> tuple[int, int, int, int]:
    """复用单例积分图计算 2×2 祖先计数，避免候选循环重复扫描百万像素。"""
    height = integral.shape[0] - 1
    width = integral.shape[1] - 1
    y_bins = _adaptive_bins(height, 2)
    x_bins = _adaptive_bins(width, 2)
    values: list[int] = []
    for target_y0, target_y1 in y_bins:
        for target_x0, target_x1 in x_bins:
            values.append(_rect_sum(integral, target_y0 - int(dy), target_y1 - int(dy), target_x0 - int(dx), target_x1 - int(dx)))
    return tuple(values)
