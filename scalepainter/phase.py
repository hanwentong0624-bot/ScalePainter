"""ScalePainter phase. Derived from the verified research implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scalepainter.phase_ancestors import _ancestor_counts_from_integral
from scalepainter.phase_stability import StableOwnershipCandidate, StableOwnershipDecision, _connected_components, _parent_core_at_fine_centers, choose_stable_ownership_phase
from scalepainter.phase_geometry import _integral_image, coverage_map_from_integral, enumerate_legal_shifts, normalize_binary_mask

@dataclass(frozen=True)
class NetStableCandidate:
    """一个第二级候选相对原生相位的稳定所有权净变化。"""
    dx: int
    dy: int
    ancestor_counts_2x2: tuple[int, int, int, int]
    ancestor_stable: bool
    core_counts: tuple[int, ...]
    component_counts: tuple[int, ...]
    stable_gain_by_pair: tuple[int, ...]
    stable_loss_by_pair: tuple[int, ...]
    stable_gain_total: int
    stable_loss_total: int
    net_stable_gain: int
    topology_safe: bool
    eligible: bool

    @property
    def shift_l1_px(self) -> int:
        return abs(int(self.dx)) + abs(int(self.dy))

@dataclass(frozen=True)
class NetStableDecision:
    """保存 E144 高置信选择、第二级扩展选择和最终两级路由。"""
    native: NetStableCandidate
    e144_decision: StableOwnershipDecision
    e144_selected: StableOwnershipCandidate
    expansion_selected: NetStableCandidate
    selected: NetStableCandidate
    selected_tier: str
    g28_route: str
    legal_one_step_count: int
    expansion_eligible_count: int

def _core_maps(integral: np.ndarray, *, dx: int, dy: int, formation_edges: Sequence[int], core_threshold: float) -> tuple[np.ndarray, ...]:
    """在一组形成尺度上离散计算完整 Token 所有权。"""
    return tuple((coverage_map_from_integral(integral, output_edge=int(edge), dx=int(dx), dy=int(dy)) >= float(core_threshold) for edge in formation_edges))

def _stable_maps(core_maps: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """返回每对相邻尺度中由完整粗祖先支持的完整细后代。"""
    return tuple((fine & _parent_core_at_fine_centers(coarse, int(fine.shape[0])) for coarse, fine in zip(core_maps[:-1], core_maps[1:])))

def _candidate(integral: np.ndarray, *, dx: int, dy: int, native_counts_2x2: tuple[int, int, int, int], native_components: Sequence[int], native_stable_maps: Sequence[np.ndarray], formation_edges: Sequence[int], core_threshold: float, routed: bool, min_stable_gain: int) -> NetStableCandidate:
    maps = _core_maps(integral, dx=int(dx), dy=int(dy), formation_edges=formation_edges, core_threshold=float(core_threshold))
    stable_maps = _stable_maps(maps)
    components = tuple((_connected_components(value) for value in maps))
    counts_2x2 = _ancestor_counts_from_integral(integral, dx=int(dx), dy=int(dy))
    gain_by_pair = tuple((int((value & ~base).sum()) for value, base in zip(stable_maps, native_stable_maps)))
    loss_by_pair = tuple((int((base & ~value).sum()) for value, base in zip(stable_maps, native_stable_maps)))
    gain_total = int(sum(gain_by_pair))
    loss_total = int(sum(loss_by_pair))
    net_gain = int(gain_total - loss_total)
    topology_safe = bool(sum(components) <= sum((int(v) for v in native_components)))
    ancestor_stable = bool(counts_2x2 == native_counts_2x2)
    eligible = bool(routed and abs(int(dx)) + abs(int(dy)) == 16 and ancestor_stable and (gain_total >= int(min_stable_gain)) and (net_gain > 0) and topology_safe)
    return NetStableCandidate(dx=int(dx), dy=int(dy), ancestor_counts_2x2=counts_2x2, ancestor_stable=ancestor_stable, core_counts=tuple((int(value.sum()) for value in maps)), component_counts=components, stable_gain_by_pair=gain_by_pair, stable_loss_by_pair=loss_by_pair, stable_gain_total=gain_total, stable_loss_total=loss_total, net_stable_gain=net_gain, topology_safe=topology_safe, eligible=eligible)

def _selection_key(candidate: NetStableCandidate) -> tuple[int, ...]:
    """优先最大净收益，再最大新增量、最小损失，方向只用于确定性破平局。"""
    return (-int(candidate.net_stable_gain), -int(candidate.stable_gain_total), int(candidate.stable_loss_total), sum(candidate.component_counts), abs(int(candidate.dy)), abs(int(candidate.dx)), int(candidate.dy), int(candidate.dx))

def choose_net_stable_ownership_phase(mask: np.ndarray, *, formation_edges: Sequence[int]=(4, 6, 8, 12), formation_weights: Sequence[float]=(4, 3, 2, 1), core_threshold: float=0.7, shift_step_px: int=16, g28_large_mask_threshold: float=0.22, min_stable_gain: int=2) -> NetStableDecision:
    """先保留 E144 高置信结果，再用净稳定所有权门扩展激活人口。"""
    if int(shift_step_px) != 16:
        raise ValueError('E145 冻结协议只允许 16 像素单步')
    binary = normalize_binary_mask(mask)
    integral = _integral_image(binary)
    routed = bool(float(binary.mean()) < float(g28_large_mask_threshold))
    native_maps = _core_maps(integral, dx=0, dy=0, formation_edges=formation_edges, core_threshold=float(core_threshold))
    native_components = tuple((_connected_components(value) for value in native_maps))
    native_stable_maps = _stable_maps(native_maps)
    native_counts_2x2 = _ancestor_counts_from_integral(integral)
    native = _candidate(integral, dx=0, dy=0, native_counts_2x2=native_counts_2x2, native_components=native_components, native_stable_maps=native_stable_maps, formation_edges=formation_edges, core_threshold=float(core_threshold), routed=routed, min_stable_gain=int(min_stable_gain))
    shifts = tuple(((int(dx), int(dy)) for dx, dy in enumerate_legal_shifts(binary, max_shift_px=int(shift_step_px), shift_step_px=int(shift_step_px)) if abs(int(dx)) + abs(int(dy)) == int(shift_step_px)))
    candidates = tuple((_candidate(integral, dx=dx, dy=dy, native_counts_2x2=native_counts_2x2, native_components=native_components, native_stable_maps=native_stable_maps, formation_edges=formation_edges, core_threshold=float(core_threshold), routed=routed, min_stable_gain=int(min_stable_gain)) for dx, dy in shifts))
    expansion_pool = [item for item in candidates if item.eligible]
    expansion = min(expansion_pool, key=_selection_key) if expansion_pool else native
    e144 = choose_stable_ownership_phase(binary, formation_edges=formation_edges, formation_weights=formation_weights, core_threshold=float(core_threshold), shift_step_px=int(shift_step_px), g28_large_mask_threshold=float(g28_large_mask_threshold))
    e144_selected = e144.same_birth
    if e144_selected.shift_l1_px > 0:
        selected = next((item for item in candidates if item.dx == e144_selected.dx and item.dy == e144_selected.dy))
        tier = 'e144_high_confidence'
    elif expansion.shift_l1_px > 0:
        selected = expansion
        tier = 'net_stable_expansion'
    else:
        selected = native
        tier = 'identity'
    return NetStableDecision(native=native, e144_decision=e144, e144_selected=e144_selected, expansion_selected=expansion, selected=selected, selected_tier=tier, g28_route='block0.10' if routed else 'e134_terminal_reclaim', legal_one_step_count=len(candidates), expansion_eligible_count=len(expansion_pool))
