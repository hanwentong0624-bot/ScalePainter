"""ScalePainter coverage carrier. Derived from the verified research implementation."""
from __future__ import annotations
import hashlib
from typing import Any, Callable, Sequence
import torch
from scalepainter.commit import E127_VARIANTS, DescendantRoutedTokenController, _coverage_2d

def _tensor_sha256(value: torch.Tensor) -> str:
    """记录 tensor 的形状、类型和连续字节，支持跨机器审计。"""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode('ascii'))
    digest.update(str(tensor.dtype).encode('ascii'))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

class FutureDescendantResidualController:
    """以未来网格路由 G20 当前尺度目标 contribution。"""

    def __init__(self, *, variant: str, token_controller: DescendantRoutedTokenController, interpolation_mode: str, block_threshold: float=0.1, minimum_future_ratio: float=2.0, coefficient_cap: float=1.0) -> None:
        self.variant = str(variant)
        if self.variant not in E127_VARIANTS:
            raise ValueError(f'E127 variant 非法：{variant}')
        if token_controller.variant != self.variant:
            raise AssertionError('E127 token/residual variant 不一致')
        self.token_controller = token_controller
        self.scale_masks = token_controller.scale_masks
        self.proxy_bits = token_controller.proxy_bits
        self.interpolation_mode = str(interpolation_mode)
        self.block_threshold = float(block_threshold)
        self.minimum_future_ratio = float(minimum_future_ratio)
        self.coefficient_cap = float(coefficient_cap)
        self.final_code_size = tuple((int(value) for value in self.scale_masks[-1].scale))
        self.final_mask_size = tuple((int(value) for value in self.scale_masks[-1].scale[-2:]))
        self.scales = [tuple((int(value) for value in item.scale)) for item in self.scale_masks]
        self.records: list[dict[str, Any]] = []
        self.final_records: list[dict[str, Any]] = []
        self.future_keep_maps_cpu: list[torch.Tensor] = []
        self.future_coefficients_cpu: list[torch.Tensor] = []

    def _codes(self, index: int, like: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """读取合法 proxy code 与相同主分支轨迹的原生 target code。"""
        proxy = self.proxy_bits[index].to(like.device)
        target = self.token_controller.target_sampled_bits_cpu[index].to(like.device)
        proxy_codes = indices_to_codes_fn(proxy, label_type='bit_label').to(like.dtype)
        target_codes = indices_to_codes_fn(target, label_type='bit_label').to(like.dtype)
        return (proxy_codes, target_codes)

    def __call__(self, *, scale_index: int, scale: Sequence[int], controlled_bits: torch.Tensor, default_codes: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        """完全复现 G20 当前尺度 coverage residual，不提前改变生成规则。"""
        del controlled_bits
        index = int(scale_index)
        if index != len(self.records):
            raise RuntimeError('E127 residual callback 尺度乱序')
        proxy_codes, target_codes = self._codes(index, default_codes, indices_to_codes_fn)
        coverage = _coverage_2d(self.scale_masks[index]).to(default_codes.device, default_codes.dtype)
        if index == 0:
            output = proxy_codes
            coefficient = torch.zeros_like(coverage)
            mode = 'g20_one_by_one_proxy'
        else:
            output = (1.0 - coverage)[None, None, None] * proxy_codes + coverage[None, None, None] * target_codes
            coefficient = coverage
            mode = 'g20_coverage_residual'
        self.records.append({'scale_index': index, 'scale': [int(value) for value in scale], 'mode': mode, 'target_coefficient_mean': float(coefficient.mean().item()), 'output_sha256': _tensor_sha256(output)})
        return output
