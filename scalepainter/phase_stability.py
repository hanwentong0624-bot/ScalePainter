"""ScalePainter phase stability. Derived from the verified research implementation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scalepainter.phase_ancestors import _ancestor_counts_from_integral, _unlock_rank
from scalepainter.phase_geometry import PhaseMetrics, _evaluate_phase_from_integral, _integral_image, coverage_map_from_integral, enumerate_legal_shifts, normalize_binary_mask

@dataclass(frozen=True)
class StableOwnershipCandidate:
    """一个单步相位在离散所有权链上的完整审计量。"""
    phase: PhaseMetrics
    ancestor_counts_2x2: tuple[int, int, int, int]
    ancestor_stable: bool
    core_counts: tuple[int, ...]
    component_counts: tuple[int, ...]
    persistence_by_pair: tuple[int, ...]
    persistence_total: int
    persistence_run: int
    improved_scale_count: int
    degraded_scale_count: int
    same_unlock: bool
    unlock_advanced: bool
    same_birth_eligible: bool
    persistent_unlock_eligible: bool

    @property
    def dx(self) -> int:
        return int(self.phase.dx)

    @property
    def dy(self) -> int:
        return int(self.phase.dy)

    @property
    def shift_l1_px(self) -> int:
        return abs(self.dx) + abs(self.dy)

@dataclass(frozen=True)
class StableOwnershipDecision:
    """保存原生相位、同出生主候选和提前出生诊断候选。"""
    native: StableOwnershipCandidate
    same_birth: StableOwnershipCandidate
    persistent_unlock: StableOwnershipCandidate
    g28_route: str
    legal_one_step_count: int
    same_birth_eligible_count: int
    persistent_unlock_eligible_count: int

def _connected_components(binary: np.ndarray) -> int:
    """以四邻域统计小型 core 网格的连通分量数量。"""
    value = np.asarray(binary, dtype=bool)
    visited = np.zeros_like(value, dtype=bool)
    count = 0
    height, width = value.shape
    for row in range(height):
        for column in range(width):
            if not value[row, column] or visited[row, column]:
                continue
            count += 1
            stack = [(row, column)]
            visited[row, column] = True
            while stack:
                y, x = stack.pop()
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < height and 0 <= xx < width and value[yy, xx] and (not visited[yy, xx]):
                        visited[yy, xx] = True
                        stack.append((yy, xx))
    return int(count)

def _parent_core_at_fine_centers(coarse: np.ndarray, fine_edge: int) -> np.ndarray:
    """把粗 core 按细 Token 中心所属的归一化粗单元投影到细网格。"""
    coarse = np.asarray(coarse, dtype=bool)
    coarse_edge = int(coarse.shape[0])
    if coarse.shape != (coarse_edge, coarse_edge):
        raise ValueError('core 网格必须为方形')
    centers = (np.arange(int(fine_edge), dtype=np.float64) + 0.5) / float(fine_edge)
    parent = np.minimum((centers * coarse_edge).astype(np.int64), coarse_edge - 1)
    return coarse[parent[:, None], parent[None, :]]

def _longest_positive_run(values: Sequence[int]) -> int:
    """返回相邻尺度所有权接力为正的最长连续边数。"""
    longest = current = 0
    for value in values:
        if int(value) > 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)

def _candidate_from_integral(integral: np.ndarray, *, dx: int, dy: int, native_phase: PhaseMetrics, native_counts_2x2: tuple[int, int, int, int], formation_edges: Sequence[int], formation_weights: Sequence[float], core_threshold: float, native_core_counts: Sequence[int] | None=None, native_component_counts: Sequence[int] | None=None, native_persistence_total: int | None=None, routed: bool) -> StableOwnershipCandidate:
    phase = _evaluate_phase_from_integral(integral, dx=int(dx), dy=int(dy), formation_edges=formation_edges, formation_weights=formation_weights, core_threshold=float(core_threshold))
    core_maps = tuple((coverage_map_from_integral(integral, output_edge=int(edge), dx=int(dx), dy=int(dy)) >= float(core_threshold) for edge in formation_edges))
    core_counts = tuple((int(value.sum()) for value in core_maps))
    component_counts = tuple((_connected_components(value) for value in core_maps))
    persistence_by_pair = tuple((int((fine & _parent_core_at_fine_centers(coarse, int(fine.shape[0]))).sum()) for coarse, fine in zip(core_maps[:-1], core_maps[1:])))
    persistence_total = int(sum(persistence_by_pair))
    counts_2x2 = _ancestor_counts_from_integral(integral, dx=int(dx), dy=int(dy))
    if native_core_counts is None:
        native_core_counts = core_counts
    if native_component_counts is None:
        native_component_counts = component_counts
    if native_persistence_total is None:
        native_persistence_total = persistence_total
    improved = sum((int(value) > int(base) for value, base in zip(core_counts, native_core_counts)))
    degraded = sum((int(value) < int(base) for value, base in zip(core_counts, native_core_counts)))
    native_rank = _unlock_rank(native_phase.earliest_core_edge, formation_edges)
    selected_rank = _unlock_rank(phase.earliest_core_edge, formation_edges)
    same_unlock = bool(selected_rank == native_rank)
    unlock_advanced = bool(selected_rank < native_rank)
    components_safe = all((int(value) <= max(int(base), 1) for value, base in zip(component_counts, native_component_counts)))
    common = bool(routed and abs(int(dx)) + abs(int(dy)) == 16 and (counts_2x2 == native_counts_2x2) and (degraded == 0) and (improved >= 2) and (persistence_total > int(native_persistence_total)) and components_safe)
    return StableOwnershipCandidate(phase=phase, ancestor_counts_2x2=counts_2x2, ancestor_stable=bool(counts_2x2 == native_counts_2x2), core_counts=core_counts, component_counts=component_counts, persistence_by_pair=persistence_by_pair, persistence_total=persistence_total, persistence_run=_longest_positive_run(persistence_by_pair), improved_scale_count=int(improved), degraded_scale_count=int(degraded), same_unlock=same_unlock, unlock_advanced=unlock_advanced, same_birth_eligible=bool(common and same_unlock), persistent_unlock_eligible=bool(common and unlock_advanced and (_longest_positive_run(persistence_by_pair) >= 2)))

def _selection_key(candidate: StableOwnershipCandidate, native: StableOwnershipCandidate) -> tuple[int, ...]:
    """只使用离散所有权量排序；方向顺序仅作确定性破平局。"""
    persistence_gain = int(candidate.persistence_total - native.persistence_total)
    total_core_gain = int(sum(candidate.core_counts) - sum(native.core_counts))
    return (-persistence_gain, -int(candidate.improved_scale_count), -total_core_gain, sum(candidate.component_counts), abs(candidate.dy), abs(candidate.dx), candidate.dy, candidate.dx)

def choose_stable_ownership_phase(mask: np.ndarray, *, formation_edges: Sequence[int]=(4, 6, 8, 12), formation_weights: Sequence[float]=(4, 3, 2, 1), core_threshold: float=0.7, shift_step_px: int=16, g28_large_mask_threshold: float=0.22) -> StableOwnershipDecision:
    """选择单步同出生相位，并保留提前出生的严格诊断列。"""
    if int(shift_step_px) != 16:
        raise ValueError('E144 冻结协议只允许 16 像素单步')
    binary = normalize_binary_mask(mask)
    integral = _integral_image(binary)
    routed = bool(float(binary.mean()) < float(g28_large_mask_threshold))
    shifts = tuple(((int(dx), int(dy)) for dx, dy in enumerate_legal_shifts(binary, max_shift_px=int(shift_step_px), shift_step_px=int(shift_step_px)) if int(dx) == 0 and int(dy) == 0 or abs(int(dx)) + abs(int(dy)) == int(shift_step_px)))
    native_phase = _evaluate_phase_from_integral(integral, dx=0, dy=0, formation_edges=formation_edges, formation_weights=formation_weights, core_threshold=float(core_threshold))
    native_counts = _ancestor_counts_from_integral(integral)
    native = _candidate_from_integral(integral, dx=0, dy=0, native_phase=native_phase, native_counts_2x2=native_counts, formation_edges=formation_edges, formation_weights=formation_weights, core_threshold=float(core_threshold), routed=routed)
    candidates = tuple((_candidate_from_integral(integral, dx=dx, dy=dy, native_phase=native_phase, native_counts_2x2=native_counts, formation_edges=formation_edges, formation_weights=formation_weights, core_threshold=float(core_threshold), native_core_counts=native.core_counts, native_component_counts=native.component_counts, native_persistence_total=native.persistence_total, routed=routed) for dx, dy in shifts if dx != 0 or dy != 0))
    same_pool = [item for item in candidates if item.same_birth_eligible]
    unlock_pool = [item for item in candidates if item.persistent_unlock_eligible]
    same_birth = min(same_pool, key=lambda item: _selection_key(item, native)) if same_pool else native
    persistent_unlock = min(unlock_pool, key=lambda item: _selection_key(item, native)) if unlock_pool else native
    return StableOwnershipDecision(native=native, same_birth=same_birth, persistent_unlock=persistent_unlock, g28_route='block0.10' if routed else 'e134_terminal_reclaim', legal_one_step_count=len(candidates), same_birth_eligible_count=len(same_pool), persistent_unlock_eligible_count=len(unlock_pool))
