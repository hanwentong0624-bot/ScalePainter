"""ScalePainter terminal. Derived from the verified research implementation."""
from __future__ import annotations
import hashlib
from typing import Any, Callable, Mapping, Sequence
import torch
import torch.nn.functional as F
from scalepainter.commit import _coverage_2d
from scalepainter.commit import DescendantRoutedResidualController
from scalepainter.coverage_carrier import FutureDescendantResidualController
E134_VARIANTS = ('e134_terminal64_early16', 'e134_terminal64_all')

def _tensor_sha256(value: torch.Tensor) -> str:
    """记录张量形状、类型和连续字节，支持服务器审计。"""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode('ascii'))
    digest.update(str(tensor.dtype).encode('ascii'))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def terminal_routing_active(*, variant: str, scale_index: int, scale: Sequence[int], final_scale_index: int, early_max_side: int=16) -> bool:
    """判断当前尺度是否启用最终网格回收。

    1x1 没有 target contribution，最终尺度也没有更细的空间归属需要查询。
    ``early16`` 的16不是经验像素半径，而是 Infinity 64x64 最终网格下
    “每个当前 token 至少对应4x4个最终后代”的机制边界。
    """
    name = str(variant)
    if name not in E134_VARIANTS:
        raise ValueError(f'E134 variant 非法：{variant}')
    index = int(scale_index)
    if index <= 0 or index >= int(final_scale_index):
        return False
    side = max((int(value) for value in scale[-2:]))
    if name == 'e134_terminal64_all':
        return True
    return side <= int(early_max_side)

def build_terminal_keep_map(final_coverage: torch.Tensor, *, threshold: float) -> torch.Tensor:
    """在最终二维网格上编译一次0/1目标归属门。"""
    coverage = final_coverage.detach().cpu().to(torch.float32)
    if coverage.ndim != 2:
        raise ValueError(f'E134 final coverage 必须二维，实际为{tuple(coverage.shape)}')
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError('E134 threshold 必须位于[0,1]')
    return coverage >= float(threshold)

class TerminalGridResidualController:
    """保持G20因果轨迹，仅用最终网格精化最终重建贡献。"""

    def __init__(self, *, variant: str, token_controller: Any, interpolation_mode: str, block_threshold: float=0.1, early_max_side: int=16, carrier_mode: str='coverage') -> None:
        self.variant = str(variant)
        if self.variant not in E134_VARIANTS:
            raise ValueError(f'E134 variant 非法：{variant}')
        self.carrier_mode = str(carrier_mode)
        if self.carrier_mode not in {'coverage', 'support_hard'}:
            raise ValueError(f'E134 carrier_mode 非法：{carrier_mode}')
        expected_token_variant = 'e127_future_descendant_route' if self.carrier_mode == 'coverage' else 'e123_supporthard_block010'
        if str(token_controller.variant) != expected_token_variant:
            raise AssertionError(f'E134 token轨迹与载体不匹配：{token_controller.variant} != {expected_token_variant}')
        self.token_controller = token_controller
        self.scale_masks = token_controller.scale_masks
        self.interpolation_mode = str(interpolation_mode)
        self.block_threshold = float(block_threshold)
        self.early_max_side = int(early_max_side)
        if abs(self.block_threshold - 0.1) > 1e-12:
            raise AssertionError('E134 冻结最终coverage阈值为0.10')
        if self.early_max_side != 16:
            raise AssertionError('E134 early范围冻结到16x16')
        self._base = FutureDescendantResidualController(variant='e127_future_descendant_route', token_controller=token_controller, interpolation_mode=self.interpolation_mode, block_threshold=self.block_threshold, minimum_future_ratio=2.0, coefficient_cap=1.0) if self.carrier_mode == 'coverage' else DescendantRoutedResidualController(token_controller=token_controller, interpolation_mode=self.interpolation_mode)
        self._g20 = self._base
        self.final_code_size = tuple((int(value) for value in self.scale_masks[-1].scale))
        self.final_mask_size = tuple((int(value) for value in self.final_code_size[-2:]))
        self.final_keep_cpu = build_terminal_keep_map(_coverage_2d(self.scale_masks[-1]), threshold=self.block_threshold)
        self.final_records: list[dict[str, Any]] = []
        self.keep_maps_final_cpu: list[torch.Tensor] = []

    def __call__(self, *, scale_index: int, scale: Sequence[int], controlled_bits: torch.Tensor, default_codes: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        """逐元素复用E127/G20的当前尺度coverage residual。"""
        return self._base(scale_index=scale_index, scale=scale, controlled_bits=controlled_bits, default_codes=default_codes, indices_to_codes_fn=indices_to_codes_fn)

    def control_final_contribution(self, *, scale_index: int, scale: Sequence[int], controlled_bits: torch.Tensor, default_final_codes: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        """用最终64x64硬门裁剪当前尺度上采样后的target-relative delta。"""
        del controlled_bits
        index = int(scale_index)
        if index != len(self.final_records):
            raise RuntimeError('E134 final contribution callback 尺度乱序')
        active = terminal_routing_active(variant=self.variant, scale_index=index, scale=scale, final_scale_index=len(self.scale_masks) - 1, early_max_side=self.early_max_side)
        if not active:
            keep_final = torch.ones(self.final_mask_size, dtype=torch.bool, device=default_final_codes.device)
            output = default_final_codes
            inside_reconstruction_error = 0.0
        else:
            proxy_codes, target_codes = self._base._codes(index, default_final_codes, indices_to_codes_fn)
            proxy_final = F.interpolate(proxy_codes, size=self.final_code_size, mode=self.interpolation_mode)
            if self.carrier_mode == 'coverage':
                coefficient = _coverage_2d(self.scale_masks[index]).to(default_final_codes.device, default_final_codes.dtype)
            else:
                coefficient = self.token_controller.plan.edit_maps[index].to(default_final_codes.device, default_final_codes.dtype)
            weighted_delta = (target_codes - proxy_codes) * coefficient[None, None, None]
            delta_final = F.interpolate(weighted_delta, size=self.final_code_size, mode=self.interpolation_mode)
            keep_final = self.final_keep_cpu.to(default_final_codes.device)
            output = proxy_final + keep_final[None, None, None].to(delta_final.dtype) * delta_final
            difference = (output.float() - default_final_codes.float()).abs()
            inside = keep_final[None, None, None].expand_as(difference)
            inside_reconstruction_error = float(difference[inside].max().item()) if bool(inside.any().item()) else 0.0
            if inside_reconstruction_error > 2e-05:
                raise AssertionError(f'E134 keep区域未复现G20 contribution：{inside_reconstruction_error}')
        if output.shape != default_final_codes.shape:
            raise AssertionError('E134 contribution shape 漂移')
        if not bool(torch.isfinite(output).all().item()):
            raise AssertionError('E134 contribution 出现NaN/Inf')
        change = output.float() - default_final_codes.float()
        self.keep_maps_final_cpu.append(keep_final.detach().cpu())
        self.final_records.append({'scale_index': index, 'scale': [int(value) for value in scale], 'routing_active': bool(active), 'terminal_scale': list(self.scale_masks[-1].scale), 'terminal_threshold': self.block_threshold, 'keep_fraction_final': float(keep_final.float().mean().item()), 'blocked_fraction_final': float((~keep_final).float().mean().item()), 'changed_max_abs': float(change.abs().max().item()), 'changed_l2': float(torch.linalg.vector_norm(change).item()), 'keep_region_reconstruction_error_max': inside_reconstruction_error, 'keep_map_sha256': _tensor_sha256(keep_final), 'output_sha256': _tensor_sha256(output)})
        return output

    def export_arrays(self) -> Mapping[str, torch.Tensor]:
        """导出各尺度coverage、共享最终门和实际启用范围。"""
        arrays: dict[str, torch.Tensor] = {'terminal_coverage': _coverage_2d(self.scale_masks[-1]), 'terminal_keep_map': self.final_keep_cpu}
        for index, item in enumerate(self.scale_masks):
            prefix = f'scale_{index:02d}'
            arrays[f'{prefix}_coverage'] = _coverage_2d(item)
            if index < len(self.keep_maps_final_cpu):
                arrays[f'{prefix}_final_keep_map'] = self.keep_maps_final_cpu[index]
        return arrays

    def finalize(self) -> dict[str, Any]:
        """验证尺度完整性、最终门一致性和所有冻结因果边界。"""
        expected = len(self.scale_masks)
        if len(self._base.records) != expected or len(self.final_records) != expected:
            raise AssertionError('E134 尺度记录不完整')
        terminal_hash = _tensor_sha256(self.final_keep_cpu)
        for row, keep in zip(self.final_records, self.keep_maps_final_cpu):
            if row['routing_active'] and _tensor_sha256(keep) != terminal_hash:
                raise AssertionError('E134 活动尺度没有共享同一最终网格门')
        return {'variant': self.variant, 'method': 'terminal_grid_exact_target_delta_reclaim', 'carrier_mode': self.carrier_mode, 'formula': 'proxy_final + terminal_keep_64 * upsample(coverage * (target-proxy))' if self.carrier_mode == 'coverage' else 'proxy_final + terminal_keep_64 * upsample(support * (target-proxy))', 'terminal_threshold': self.block_threshold, 'early_max_side': self.early_max_side, 'full_source_mask_interior_received': False, 'attention_modified': False, 'logits_modified': False, 'kv_cache_modified': False, 'target_budget_recycled': False, 'current_g20_residual_unchanged': self.carrier_mode == 'coverage', 'current_support_hard_residual_unchanged': self.carrier_mode == 'support_hard', 'records': list(self._base.records), 'final_contribution_records': list(self.final_records)}
