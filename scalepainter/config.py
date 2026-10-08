"""Small, explicit configuration objects for inference and paper ablations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodConfig:
    coverage: bool = True
    reclamation: bool = True
    routed_terminal: bool = True
    mask_size_prompt: bool = True
    phase_alignment: bool = True
    terminal_area_threshold: float = 0.22
    block_threshold: float = 0.10
    early_max_side: int = 16
    core_threshold: float = 0.7


@dataclass(frozen=True)
class SamplingConfig:
    cfg: float = 4.0
    tau: float = 0.5
    top_k: int = 900
    top_p: float = 0.97


# Eight distinct configurations; the final two aliases do not add new runs.
VARIANTS = {
    'A0': MethodConfig(coverage=False, reclamation=False, routed_terminal=False, mask_size_prompt=False, phase_alignment=False),
    'A1': MethodConfig(reclamation=False, routed_terminal=False, mask_size_prompt=False, phase_alignment=False),
    'A2': MethodConfig(mask_size_prompt=False, phase_alignment=False),
    'A3': MethodConfig(phase_alignment=False),
    'A4': MethodConfig(mask_size_prompt=False),
    'A_all': MethodConfig(),
    'A_wo1': MethodConfig(coverage=False),
    'A_wo2': MethodConfig(reclamation=False),
}
ALIASES = {'A_wo3': 'A4', 'A_wo4': 'A3'}


def get_variant(name: str) -> MethodConfig:
    return VARIANTS[ALIASES.get(name, name)]


def mask_stratum(ratio: float, dataset: str) -> str:
    if dataset not in {'EditBench', 'BrushBench', 'generic'}:
        raise ValueError(f'Unknown dataset: {dataset}')
    upper = 0.30 if dataset == 'BrushBench' else 0.25
    return 'small' if ratio < 0.10 else 'medium' if ratio < upper else 'large'


def build_controllers(config, proxy_bits, scale_masks, schedule, interpolation_mode, original_mask_ratio):
    from .commit import DescendantRoutedTokenController, DescendantRoutedResidualController
    from .terminal import TerminalGridResidualController
    from .no_reclamation import NoReclamationResidualController

    terminal = config.routed_terminal and original_mask_ratio >= config.terminal_area_threshold
    # A_wo2 keeps the complete parent's route and initial-scale convention.
    no_slc_parent = not config.reclamation and config.routed_terminal
    if not config.coverage:
        token_variant = 'e123_supporthard_block010' if config.reclamation else 'e123_supporthard_no_route'
    elif terminal:
        token_variant = 'e127_future_descendant_route'
    elif config.reclamation or no_slc_parent:
        token_variant = 'e124_g20_block010'
    else:
        token_variant = 'e124_g20_no_route_ref'
    token = DescendantRoutedTokenController(
        variant=token_variant, proxy_bits=proxy_bits, scale_masks=scale_masks,
        edit_threshold=0.0,
        block_threshold=0.0 if token_variant == 'e124_g20_no_route_ref' else config.block_threshold,
        callback_scales=schedule,
    )
    if terminal:
        residual = TerminalGridResidualController(
            variant='e134_terminal64_early16', token_controller=token,
            interpolation_mode=interpolation_mode,
            block_threshold=config.block_threshold, early_max_side=config.early_max_side,
            carrier_mode='coverage' if config.coverage else 'support_hard',
        )
    else:
        residual = DescendantRoutedResidualController(token_controller=token, interpolation_mode=interpolation_mode)
    if no_slc_parent:
        residual = NoReclamationResidualController(residual, terminal=terminal)
    return token, residual, 'terminal' if terminal else 'local'
