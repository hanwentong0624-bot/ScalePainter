"""ScalePainter commit. Derived from the verified research implementation."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import torch
import torch.nn.functional as F
E122_VARIANTS = ('e122_g20_coverage_ref', 'e122_hard050_no_route', 'e122_hard050_block000', 'e122_hard050_block010')
E123_VARIANTS = ('e123_g20_coverage_ref', 'e123_hard050_block010_ref', 'e123_supporthard_no_route', 'e123_supporthard_block010')
E124_VARIANTS = ('e124_g20_no_route_ref', 'e124_g20_block000', 'e124_g20_block005', 'e124_g20_block010', 'e124_g20_block015', 'e124_g20_block020')
E127_VARIANTS = ('e127_future_descendant_route', 'e127_future_route_budget_recycle')
E124_BLOCK_THRESHOLDS = {'e124_g20_no_route_ref': None, 'e124_g20_block000': 0.0, 'e124_g20_block005': 0.05, 'e124_g20_block010': 0.1, 'e124_g20_block015': 0.15, 'e124_g20_block020': 0.2}
_ALL_VARIANTS = E122_VARIANTS + E123_VARIANTS + E124_VARIANTS + E127_VARIANTS
_G20_REFERENCE_VARIANTS = {'e122_g20_coverage_ref', 'e123_g20_coverage_ref', *E124_VARIANTS, *E127_VARIANTS}
_BLOCK010_VARIANTS = {'e122_hard050_block010', 'e123_hard050_block010_ref', 'e123_supporthard_block010'}
_NO_ROUTE_VARIANTS = {'e122_g20_coverage_ref', 'e122_hard050_no_route', 'e123_g20_coverage_ref', 'e123_supporthard_no_route', 'e124_g20_no_route_ref'}

def e124_variant_block_threshold(variant: str) -> float | None:
    """返回 E124 变体冻结的后代回收阈值；参考列返回 ``None``。"""
    name = str(variant)
    if name not in E124_BLOCK_THRESHOLDS:
        raise ValueError(f'未知 E124 variant：{variant}')
    return E124_BLOCK_THRESHOLDS[name]

def _tensor_sha256(value: torch.Tensor) -> str:
    """记录 tensor 的形状、类型和连续字节，便于逐尺度审计。"""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode('ascii'))
    digest.update(str(tensor.dtype).encode('ascii'))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def _coverage_2d(item: Any) -> torch.Tensor:
    """把 ScaleMask coverage 规范化为 CPU float32 二维张量。"""
    coverage = item.coverage.detach().cpu().to(dtype=torch.float32)
    if coverage.ndim != 2:
        height, width = (int(value) for value in item.scale[1:3])
        coverage = coverage.reshape(height, width)
    return coverage.clamp(0.0, 1.0)

@dataclass(frozen=True)
class DescendantRoutingPlan:
    """保存推理前由 mask 金字塔编译出的全部离散空间门。"""
    edit_maps: tuple[torch.Tensor, ...]
    keep_maps_next: tuple[torch.Tensor | None, ...]
    records: tuple[dict[str, Any], ...]

def build_descendant_routing_plan(*, variant: str, scale_masks: Sequence[Any], edit_threshold: float=0.5, block_threshold: float=0.1) -> DescendantRoutingPlan:
    """仅根据 coverage 构造当前提交门和下一尺度后代保留门。

    ``block000`` 使用严格 ``coverage > 0`` 作为保留条件，只阻断确定 known；
    ``block010`` 使用 ``coverage >= 0.1``。阈值比较在 CPU float32 上一次性完成，
    生成时不读取图像内容或动态语义分数。
    """
    name = str(variant)
    if name not in _ALL_VARIANTS:
        raise ValueError(f'E122/E123 未知 variant：{variant}')
    if not scale_masks:
        raise ValueError('E122 scale_masks 不能为空')
    expected_edit_threshold = 0.0 if name in E124_VARIANTS or name in E127_VARIANTS or name in {'e123_supporthard_no_route', 'e123_supporthard_block010', 'e123_g20_coverage_ref'} else 0.5
    if abs(float(edit_threshold) - expected_edit_threshold) > 1e-12:
        raise AssertionError(f'{name} 正式实验冻结 edit_threshold={expected_edit_threshold}')
    if name in E124_VARIANTS:
        expected_block = e124_variant_block_threshold(name)
        if expected_block is not None and abs(float(block_threshold) - float(expected_block)) > 1e-12:
            raise AssertionError(f'{name} 正式实验冻结 block_threshold={expected_block}')
    elif abs(float(block_threshold) - 0.1) > 1e-12:
        raise AssertionError('E122a/E123a 正式实验冻结 block_threshold=0.1')
    coverage_maps = [_coverage_2d(item) for item in scale_masks]
    edit_maps: list[torch.Tensor] = []
    keep_maps: list[torch.Tensor | None] = []
    records: list[dict[str, Any]] = []
    for index, coverage in enumerate(coverage_maps):
        edit = torch.zeros_like(coverage, dtype=torch.bool)
        if index > 0:
            edit = coverage > float(edit_threshold)
        keep: torch.Tensor | None = None
        if index < len(coverage_maps) - 1:
            next_coverage = coverage_maps[index + 1]
            if name in {'e122_hard050_block000', 'e124_g20_block000'}:
                keep = next_coverage > 0.0
            elif name in E127_VARIANTS:
                keep = next_coverage >= float(block_threshold)
            elif name in E124_VARIANTS and name not in _NO_ROUTE_VARIANTS:
                keep = next_coverage >= float(block_threshold)
            elif name in _BLOCK010_VARIANTS:
                keep = next_coverage >= float(block_threshold)
            else:
                keep = torch.ones_like(next_coverage, dtype=torch.bool)
        records.append({'scale_index': int(index), 'scale': [int(value) for value in scale_masks[index].scale], 'edit_threshold': float(edit_threshold), 'edit_token_count': int(edit.sum().item()), 'coverage_equal_050_count': int((coverage == 0.5).sum().item()), 'coverage_equal_edit_threshold_count': int((coverage == float(edit_threshold)).sum().item()), 'next_keep_token_count': None if keep is None else int(keep.sum().item()), 'next_block_token_count': None if keep is None else int((~keep).sum().item()), 'next_block_threshold': None if keep is None or name in _NO_ROUTE_VARIANTS else 0.0 if name.endswith('block000') else float(block_threshold)})
        edit_maps.append(edit)
        keep_maps.append(keep)
    if bool(edit_maps[0].any().item()):
        raise AssertionError('E122 1x1 不得提交 target')
    if name in E127_VARIANTS or name in _BLOCK010_VARIANTS or (name in E124_VARIANTS and name not in _NO_ROUTE_VARIANTS):
        for index in range(len(coverage_maps) - 1):
            strict_known = coverage_maps[index + 1] <= 0.0
            if bool((strict_known & keep_maps[index]).any().item()):
                raise AssertionError('E122 block010 未包含全部 exact-known 后代')
    return DescendantRoutingPlan(edit_maps=tuple(edit_maps), keep_maps_next=tuple(keep_maps), records=tuple(records))

class DescendantRoutedTokenController:
    """按 G20 或 coverage>0.5 日程重组当前尺度 hard bits。"""

    def __init__(self, *, variant: str, proxy_bits: Sequence[torch.Tensor], scale_masks: Sequence[Any], edit_threshold: float=0.5, block_threshold: float=0.1, callback_scales: Sequence[Sequence[int]] | None=None) -> None:
        self.variant = str(variant)
        if self.variant not in _ALL_VARIANTS:
            raise ValueError(f'E122/E123 token variant 非法：{variant}')
        self.edit_threshold = float(edit_threshold)
        self.block_threshold = float(block_threshold)
        self.proxy_bits = [value.detach().cpu().clone() for value in proxy_bits]
        self.scale_masks = list(scale_masks)
        if len(self.proxy_bits) != len(self.scale_masks):
            raise ValueError('E122 proxy bits 与 scale masks 数量不一致')
        self.callback_scales = [tuple((int(value) for value in scale)) for scale in callback_scales] if callback_scales is not None else [tuple((int(value) for value in item.scale)) for item in self.scale_masks]
        if len(self.callback_scales) != len(self.scale_masks):
            raise ValueError('E122 callback scales 与 scale masks 数量不一致')
        self.plan = build_descendant_routing_plan(variant=self.variant, scale_masks=self.scale_masks, edit_threshold=edit_threshold, block_threshold=block_threshold)
        self.target_sampled_bits_cpu: list[torch.Tensor] = []
        self.controlled_bits_cpu: list[torch.Tensor] = []
        self.mixed_bits_cpu = self.controlled_bits_cpu
        self.provenance_maps_cpu: list[torch.Tensor] = []
        self.records: list[dict[str, Any]] = []

    def __call__(self, *, scale_index: int, scale: Sequence[int], sampled_bits: torch.Tensor) -> torch.Tensor:
        """选择完整 target/proxy bit-vector，不进行逐 bit 混合。"""
        index = int(scale_index)
        if index != len(self.records):
            raise RuntimeError('E122 token callback 尺度乱序')
        item = self.scale_masks[index]
        if tuple((int(value) for value in scale)) != self.callback_scales[index]:
            raise ValueError('E122 token callback 尺度漂移')
        proxy = self.proxy_bits[index].to(sampled_bits.device, sampled_bits.dtype)
        if proxy.shape != sampled_bits.shape:
            raise ValueError('E122 proxy/target bits shape 不一致')
        if tuple((int(value) for value in sampled_bits.shape[2:4])) != tuple((int(value) for value in item.scale[1:])):
            raise ValueError('E122 sampled bits 与 VAE mask 网格不一致')
        if self.variant in _G20_REFERENCE_VARIANTS:
            target_region = torch.zeros_like(item.support, dtype=torch.bool) if index == 0 else item.support.detach().cpu().bool()
        else:
            target_region = self.plan.edit_maps[index]
        selector = target_region.to(sampled_bits.device)[None, None, :, :, None]
        selector = selector.expand_as(sampled_bits)
        controlled = torch.where(selector, sampled_bits, proxy)
        provenance = torch.ones_like(target_region, dtype=torch.uint8)
        coverage = _coverage_2d(item)
        provenance[coverage > 0] = 2
        provenance[target_region] = 3
        if bool(((provenance == 4) | (provenance == 5)).any().item()):
            raise AssertionError('E122 出现非法 provenance code 4/5')
        self.target_sampled_bits_cpu.append(sampled_bits.detach().cpu().clone())
        self.controlled_bits_cpu.append(controlled.detach().cpu().clone())
        self.provenance_maps_cpu.append(provenance)
        record = dict(self.plan.records[index])
        record.update({'target_token_count_runtime': int(target_region.sum().item()), 'proxy_token_count_runtime': int(target_region.numel() - target_region.sum().item()), 'target_region_sha256': _tensor_sha256(target_region), 'target_sampled_bits_sha256': _tensor_sha256(sampled_bits), 'controlled_bits_sha256': _tensor_sha256(controlled)})
        self.records.append(record)
        return controlled

    def finalize(self) -> dict[str, Any]:
        """完成 no-leak、1x1 和完整 Token 审计。"""
        if len(self.records) != len(self.scale_masks):
            raise AssertionError('E122 token controller 未覆盖全部尺度')
        return {'variant': self.variant, 'method': 'double_threshold_descendant_routed_whole_token_commit', 'full_source_mask_interior_received': False, 'target_proxy_bitwise_mixing': False, 'one_by_one_target_count': int(self.plan.edit_maps[0].sum().item()), 'records': list(self.records)}

class DescendantRoutedResidualController:
    """提交当前完整 code，并在最终 latent 上执行预编译后代路由。"""

    def __init__(self, *, token_controller: DescendantRoutedTokenController, interpolation_mode: str) -> None:
        self.token_controller = token_controller
        self.variant = token_controller.variant
        self.scale_masks = token_controller.scale_masks
        self.proxy_bits = token_controller.proxy_bits
        self.interpolation_mode = str(interpolation_mode)
        self.final_size = tuple((int(value) for value in self.scale_masks[-1].scale))
        self.records: list[dict[str, Any]] = []
        self.final_records: list[dict[str, Any]] = []
        self.keep_maps_final_cpu: list[torch.Tensor] = []
        self.edit_maps_cpu = list(token_controller.plan.edit_maps)

    def _codes(self, index: int, like: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """读取同尺度合法 proxy code 和主分支原生 target code。"""
        proxy = self.proxy_bits[index].to(like.device)
        target = self.token_controller.target_sampled_bits_cpu[index].to(like.device)
        proxy_codes = indices_to_codes_fn(proxy, label_type='bit_label').to(like.dtype)
        target_codes = indices_to_codes_fn(target, label_type='bit_label').to(like.dtype)
        return (proxy_codes, target_codes)

    def __call__(self, *, scale_index: int, scale: Sequence[int], controlled_bits: torch.Tensor, default_codes: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        """构造本尺度 code；只有 G20 reference 使用 coverage 连续 residual。"""
        del controlled_bits
        index = int(scale_index)
        if index != len(self.records):
            raise RuntimeError('E122 residual callback 尺度乱序')
        item = self.scale_masks[index]
        proxy_codes, target_codes = self._codes(index, default_codes, indices_to_codes_fn)
        coverage = _coverage_2d(item).to(default_codes.device, default_codes.dtype)
        if self.variant in _G20_REFERENCE_VARIANTS:
            if index == 0:
                output = proxy_codes
                coefficient = torch.zeros_like(coverage)
                mode = 'g20_one_by_one_proxy'
            else:
                output = (1.0 - coverage)[None, None, None] * proxy_codes + coverage[None, None, None] * target_codes
                coefficient = coverage
                mode = 'g20_coverage_residual'
        else:
            edit = self.token_controller.plan.edit_maps[index].to(default_codes.device)
            output = torch.where(edit[None, None, None], target_codes, proxy_codes)
            coefficient = edit.to(dtype=default_codes.dtype)
            mode = 'supporthard_target_or_proxy_code' if self.token_controller.edit_threshold == 0.0 else 'hard050_target_or_proxy_code'
        if not bool(torch.isfinite(output).all().item()):
            raise AssertionError('E122 residual 出现 NaN/Inf')
        self.records.append({'scale_index': index, 'scale': [int(value) for value in scale], 'mode': mode, 'target_coefficient_mean': float(coefficient.float().mean().item()), 'fractional_coefficient_count': int(((coefficient > 0) & (coefficient < 1)).sum().item()), 'output_sha256': _tensor_sha256(output)})
        return output

    def control_final_contribution(self, *, scale_index: int, scale: Sequence[int], controlled_bits: torch.Tensor, default_final_codes: torch.Tensor, indices_to_codes_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        """在 contribution 写入累计 latent 前阻断低 coverage 后代。

        输出以完整 proxy contribution 为基底，再加经过 0/1 后代门的目标增量。
        该公式不读取当前或未来生成结果；``keep`` 在推理开始前已经由 mask 编译。
        """
        del controlled_bits
        index = int(scale_index)
        if index != len(self.final_records):
            raise RuntimeError('E122 final contribution callback 尺度乱序')
        if self.variant in _NO_ROUTE_VARIANTS or index == len(self.scale_masks) - 1:
            keep_final = torch.ones(default_final_codes.shape[-3:], device=default_final_codes.device, dtype=torch.bool)
            output = default_final_codes
            changed = False
        else:
            proxy_codes, target_codes = self._codes(index, default_final_codes, indices_to_codes_fn)
            if self.variant in E124_VARIANTS:
                coverage = _coverage_2d(self.scale_masks[index]).to(default_final_codes.device, default_final_codes.dtype)
                delta_codes = (target_codes - proxy_codes) * coverage[None, None, None]
                edit_active = bool((coverage > 0).any().item())
            else:
                edit = self.token_controller.plan.edit_maps[index].to(default_final_codes.device)
                delta_codes = (target_codes - proxy_codes) * edit[None, None, None]
                edit_active = bool(edit.any().item())
            proxy_final = F.interpolate(proxy_codes, size=self.final_size, mode=self.interpolation_mode)
            delta_final = F.interpolate(delta_codes, size=self.final_size, mode=self.interpolation_mode)
            keep_next = self.token_controller.plan.keep_maps_next[index]
            if keep_next is None:
                raise AssertionError('E122 非最终尺度缺少后代 keep map')
            keep_final = F.interpolate(keep_next.to(default_final_codes.device, torch.float32)[None, None, None], size=self.final_size, mode='nearest').to(dtype=torch.bool)[0, 0, 0]
            output = proxy_final + keep_final[None, None, None].to(delta_final.dtype) * delta_final
            changed = not bool(keep_final.all().item()) and edit_active
        if output.shape != default_final_codes.shape:
            raise AssertionError('E122 routed contribution shape 漂移')
        if not bool(torch.isfinite(output).all().item()):
            raise AssertionError('E122 routed contribution 出现 NaN/Inf')
        delta = output.float() - default_final_codes.float()
        self.keep_maps_final_cpu.append(keep_final.detach().cpu())
        self.final_records.append({'scale_index': index, 'scale': [int(value) for value in scale], 'routing_active': bool(changed), 'keep_fraction_final': float(keep_final.float().mean().item()), 'blocked_fraction_final': float((~keep_final).float().mean().item()), 'changed_max_abs': float(delta.abs().max().item()), 'changed_l2': float(torch.linalg.vector_norm(delta).item()), 'keep_map_sha256': _tensor_sha256(keep_final), 'output_sha256': _tensor_sha256(output)})
        return output

    def export_arrays(self) -> Mapping[str, torch.Tensor]:
        """导出 coverage、当前提交门、下一尺度门和最终 latent 门。"""
        arrays: dict[str, torch.Tensor] = {}
        for index, item in enumerate(self.scale_masks):
            prefix = f'scale_{index:02d}'
            arrays[f'{prefix}_coverage'] = _coverage_2d(item)
            arrays[f'{prefix}_edit_map'] = self.edit_maps_cpu[index]
            keep_next = self.token_controller.plan.keep_maps_next[index]
            if keep_next is not None:
                arrays[f'{prefix}_next_keep_map'] = keep_next
            if index < len(self.keep_maps_final_cpu):
                arrays[f'{prefix}_final_keep_map'] = self.keep_maps_final_cpu[index]
        return arrays

    def finalize(self) -> dict[str, Any]:
        """验证 whole-token、后代包含关系和默认参考恒等性。"""
        expected = len(self.scale_masks)
        if len(self.records) != expected or len(self.final_records) != expected:
            raise AssertionError('E122 residual/final contribution 尺度记录不完整')
        hard_variant = self.variant not in _G20_REFERENCE_VARIANTS
        if hard_variant and any((int(row['fractional_coefficient_count']) != 0 for row in self.records)):
            raise AssertionError('E122 hard050 列出现连续 residual 系数')
        formula = 'proxy + descendant_keep * upsample(coverage * (target-proxy))' if self.variant in E124_VARIANTS else 'proxy + descendant_keep * upsample(edit * (target-proxy))'
        return {'variant': self.variant, 'method': 'precompiled_descendant_routed_contribution', 'formula': formula, 'edit_threshold': self.token_controller.edit_threshold, 'block_threshold': e124_variant_block_threshold(self.variant) if self.variant in E124_VARIANTS else self.token_controller.block_threshold if self.variant in _BLOCK010_VARIANTS else None, 'full_source_mask_interior_received': False, 'attention_modified': False, 'logits_modified': False, 'continuous_target_proxy_mix_used': not hard_variant, 'records': list(self.records), 'final_contribution_records': list(self.final_records)}
