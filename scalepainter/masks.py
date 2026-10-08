"""ScalePainter masks. Derived from the verified research implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class ScaleMask:
    """保存一个 Infinity 尺度对应的面积覆盖率、core 与 support mask。"""
    scale_index: int
    scale: tuple[int, int, int]
    coverage: torch.Tensor
    core: torch.Tensor
    support: torch.Tensor

    @property
    def known(self) -> torch.Tensor:
        """返回与编辑区域完全不相交、可作为可信源上下文的 token mask。

        ``support`` 的定义是 coverage 大于零，因此它的布尔补集正好是
        coverage 等于零的已知区域。集中在这里定义可以避免各实验自行推导时
        出现字段名漂移，同时不会引入或读取 mask 内的代理内容。
        """
        return ~self.support

    @property
    def core_tokens(self) -> int:
        """返回当前尺度明确位于编辑区域内部的 token 数。"""
        return int(self.core.sum().item())

    @property
    def support_tokens(self) -> int:
        """返回当前尺度与编辑区域发生任意相交的 token 数。"""
        return int(self.support.sum().item())

def _normalize_pixel_mask(pixel_mask: torch.Tensor) -> torch.Tensor:
    """把二维或单 batch mask 规范为 ``[1,1,H,W]`` 的 float tensor。"""
    if pixel_mask.ndim == 2:
        normalized = pixel_mask[None, None]
    elif pixel_mask.ndim == 3 and pixel_mask.shape[0] == 1:
        normalized = pixel_mask[:, None]
    elif pixel_mask.ndim == 4 and pixel_mask.shape[:2] == (1, 1):
        normalized = pixel_mask
    else:
        raise ValueError(f'pixel_mask 必须是 [H,W]、[1,H,W] 或 [1,1,H,W]，实际为 {tuple(pixel_mask.shape)}')
    normalized = normalized.float()
    if not bool(torch.isfinite(normalized).all().item()):
        raise ValueError('pixel_mask 含 NaN 或 Inf')
    if float(normalized.min().item()) < 0.0 or float(normalized.max().item()) > 1.0:
        raise ValueError('pixel_mask 必须位于 [0,1]')
    return normalized

def build_scale_masks(pixel_mask: torch.Tensor, scale_schedule: Sequence[Sequence[int]], core_threshold: float) -> list[ScaleMask]:
    """用面积池化构造所有尺度的 core/support mask。

    ``coverage`` 表示一个 token 单元被像素 mask 覆盖的比例。core 使用稳定阈值，
    support 只判断是否有任意正覆盖。E1a 只允许 core 生成，support 暂时仅用于记录
    和后续 E1b 的 inner/outer ring 构造。
    """
    if not 0.0 < float(core_threshold) <= 1.0:
        raise ValueError('core_threshold 必须位于 (0,1]')
    normalized = _normalize_pixel_mask(pixel_mask)
    masks: list[ScaleMask] = []
    for scale_index, raw_scale in enumerate(scale_schedule):
        if len(raw_scale) != 3:
            raise ValueError(f'scale 必须为 (t,h,w)，实际为 {raw_scale}')
        temporal, height, width = (int(value) for value in raw_scale)
        if temporal != 1:
            raise ValueError('E1a 当前只支持图像尺度，即 temporal=1')
        if height <= 0 or width <= 0:
            raise ValueError(f'非法空间尺度：{raw_scale}')
        coverage = F.adaptive_avg_pool2d(normalized, output_size=(height, width))[0, 0]
        masks.append(ScaleMask(scale_index=scale_index, scale=(temporal, height, width), coverage=coverage.detach().cpu(), core=(coverage >= float(core_threshold)).detach().cpu(), support=(coverage > 0.0).detach().cpu()))
    if not masks:
        raise ValueError('scale_schedule 不能为空')
    return masks
