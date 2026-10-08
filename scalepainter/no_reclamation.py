"""ScalePainter no reclamation. Derived from the verified research implementation."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from scalepainter.commit import _coverage_2d, _tensor_sha256
from scalepainter.terminal import terminal_routing_active

class NoReclamationResidualController:

    def __init__(self, parent, *, terminal: bool):
        self.parent = parent
        self.terminal = bool(terminal)
        self.base = parent._base if terminal else parent
        self.scale_masks = parent.scale_masks
        self.token_controller = parent.token_controller
        self.variant = 'g41_all_reclamation_off'
        self.final_records = []
        self.keep_maps_final_cpu = []
        if terminal:
            assert parent.carrier_mode == 'coverage'
        else:
            assert parent.variant == 'e124_g20_block010'

    def __call__(self, **kwargs):
        return self.parent(**kwargs)

    def control_final_contribution(self, *, scale_index, scale, controlled_bits, default_final_codes, indices_to_codes_fn):
        del controlled_bits
        index = int(scale_index)
        if index != len(self.final_records):
            raise RuntimeError('G41 final contribution callback out of order')
        last = len(self.scale_masks) - 1
        explicit_delta = terminal_routing_active(variant=self.parent.variant, scale_index=index, scale=scale, final_scale_index=last, early_max_side=self.parent.early_max_side) if self.terminal else index != last
        if explicit_delta:
            proxy, target = self.base._codes(index, default_final_codes, indices_to_codes_fn)
            coverage = _coverage_2d(self.scale_masks[index]).to(default_final_codes.device, default_final_codes.dtype)
            size = tuple((int(x) for x in self.scale_masks[-1].scale))
            output = F.interpolate(proxy, size=size, mode=self.parent.interpolation_mode)
            output = output + F.interpolate((target - proxy) * coverage[None, None, None], size=size, mode=self.parent.interpolation_mode)
        else:
            output = default_final_codes
        assert output.shape == default_final_codes.shape
        assert bool(torch.isfinite(output).all())
        keep = torch.ones(output.shape[-2:], dtype=torch.bool)
        self.keep_maps_final_cpu.append(keep)
        self.final_records.append({'scale_index': index, 'scale': list(scale), 'routing_active': False, 'local_reclamation_enabled': False, 'terminal_reclamation_enabled': False, 'keep_fraction_final': 1.0, 'blocked_fraction_final': 0.0, 'parent_explicit_delta_branch': bool(explicit_delta), 'keep_map_sha256': _tensor_sha256(keep), 'output_sha256': _tensor_sha256(output), 'changed_max_abs': float((output.float() - default_final_codes.float()).abs().max())})
        return output

    def export_arrays(self):
        arrays = {}
        for i, mask in enumerate(self.scale_masks):
            prefix = f'scale_{i:02d}'
            arrays[prefix + '_coverage'] = _coverage_2d(mask)
            arrays[prefix + '_edit_map'] = self.token_controller.plan.edit_maps[i]
            if i < len(self.keep_maps_final_cpu):
                arrays[prefix + '_final_keep_map'] = self.keep_maps_final_cpu[i]
            if i + 1 < len(self.scale_masks):
                arrays[prefix + '_next_keep_map'] = torch.ones_like(_coverage_2d(self.scale_masks[i + 1]), dtype=torch.bool)
        arrays['terminal_keep_map'] = torch.ones_like(_coverage_2d(self.scale_masks[-1]), dtype=torch.bool)
        return arrays

    def finalize(self):
        expected = len(self.scale_masks)
        assert len(self.base.records) == len(self.final_records) == expected
        assert all((bool(k.all()) for k in self.keep_maps_final_cpu))
        assert not self.parent.final_records, 'Parent reclamation callback must never execute'
        return {'method': 'g41_all_reclamation_off', 'variant': self.variant, 'parent_branch': 'terminal' if self.terminal else 'local', 'formula': 'parent contribution with every post-upsampling spatial gate L=1', 'local_reclamation_enabled': False, 'terminal_reclamation_enabled': False, 'initial_target_admission_convention_preserved': True, 'current_g20_residual_unchanged': True, 'full_source_mask_interior_received': False, 'records': list(self.base.records), 'final_contribution_records': list(self.final_records)}
