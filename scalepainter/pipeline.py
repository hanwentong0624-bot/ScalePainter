"""One-trajectory inference with explicit geometry, sampling and audit outputs."""
from dataclasses import asdict
import random
import time

import numpy as np
import torch

from .backbone import encode_prompt
from .config import SamplingConfig, build_controllers, get_variant, mask_stratum
from .geometry import prepare_input, restore_native
from .masks import build_scale_masks
from .phase import choose_net_stable_ownership_phase
from .phase_geometry import shift_mask_zero_fill
from .prompt import add_mask_size_word
from .proxy import build_nearest_proxy
from .translation import inverse_shift_with_source_fill, shift_image_reflect


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FinalLatent:
    def __init__(self, final_index):
        self.final_index = final_index
        self.latent = None
        self.indices = []

    def __call__(self, *, scale_index, scale, summed_codes):
        if scale_index != len(self.indices):
            raise RuntimeError('Out-of-order accumulation callback')
        self.indices.append(scale_index)
        if scale_index == self.final_index:
            # Match the frozen deferred decode's float32 CPU round trip.
            self.latent = summed_codes.detach().float().squeeze(-3).cpu().clone()


class ScalePainter:
    def __init__(self, model, vae, tokenizer, text_encoder):
        self.model, self.vae = model, vae
        self.tokenizer, self.text_encoder = tokenizer, text_encoder
        self.device = next(vae.parameters()).device

    @classmethod
    def from_pretrained(cls, **kwargs):
        from .backbone import load_models
        return cls(*load_models(**kwargs))

    @torch.inference_mode()
    def __call__(self, rgb, mask, prompt, *, seed=0, variant='A_all', geometry_mode='generic',
                 dataset='generic', stratum=None, sampling=None, return_arrays=False):
        from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

        if not str(prompt).strip():
            raise ValueError('Prompt cannot be empty')
        method = get_variant(variant)
        sampling = sampling or SamplingConfig()
        safe_u8, placement, geometry, input_audit = prepare_input(rgb, mask, mode=geometry_mode)
        source = safe_u8.astype(np.float32) / 255.0
        ratio = float(placement.mean())
        phase = choose_net_stable_ownership_phase(placement) if method.phase_alignment else None
        dx, dy = (phase.selected.dx, phase.selected.dy) if phase else (0, 0)
        proxy = build_nearest_proxy(source, placement).image
        proxy_model = shift_image_reflect(proxy, dx, dy)
        model_mask = shift_mask_zero_fill(placement, dx, dy)
        if int(model_mask.sum()) != int(placement.sum()):
            raise AssertionError('Phase alignment cropped the mask')
        # Upstream also serves video; the frozen image runner sets temporal=1.
        schedule = [(1, int(h), int(w)) for _, h, w in dynamic_resolution_h_w[1.0]['1M']['scales']]
        vae_schedule = [(t, 2*h, 2*w) for t, h, w in schedule]
        tensor = torch.from_numpy(np.ascontiguousarray(proxy_model)).permute(2, 0, 1)
        tensor = (tensor.float().unsqueeze(0).to(self.device) * 2.0 - 1.0).contiguous()
        _, _, _, encoded, _, _ = self.vae.encode(tensor, scale_schedule=vae_schedule)
        proxy_bits = [bits.detach().cpu().clone() for bits in encoded]
        scale_masks = build_scale_masks(torch.from_numpy(model_mask.astype(np.float32)), vae_schedule,
                                        core_threshold=method.core_threshold)
        token, residual, route = build_controllers(method, proxy_bits, scale_masks, schedule,
                                                   str(self.vae.quantizer.z_interplote_up), ratio)
        original_prompt = str(prompt).strip()
        stratum = stratum or mask_stratum(ratio, dataset)
        model_prompt = add_mask_size_word(original_prompt, stratum).expanded_prompt if method.mask_size_prompt else original_prompt
        condition = encode_prompt(self.tokenizer, self.text_encoder, model_prompt)
        observer = FinalLatent(len(schedule)-1)
        set_seed(seed)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=True):
                self.model.autoregressive_infer_cfg(
                    vae=self.vae, scale_schedule=schedule, label_B_or_BLT=condition,
                    g_seed=seed, B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                    cfg_sc=3, cfg_list=[sampling.cfg]*len(schedule), tau_list=[sampling.tau]*len(schedule),
                    top_k=sampling.top_k, top_p=sampling.top_p, returns_vemb=1,
                    ratio_Bl1=None, gumbel=0, norm_cfg=False, cfg_exp_k=0.0,
                    cfg_insertion_layer=[0], vae_type=14, softmax_merge_topk=-1,
                    ret_img=False, trunk_scale=1000, gt_leak=0, gt_ls_Bl=None,
                    inference_mode=True, sampling_per_bits=1,
                    scale_reassembly_fn=token, scale_code_reassembly_fn=residual,
                    scale_accumulation_diagnostics_fn=observer,
                )
        finally:
            # Failed batches must not retain an incomplete trajectory's KV cache.
            for block in self.model.unregistered_blocks:
                block.sa.kv_caching(False)
        torch.cuda.synchronize(self.device)
        generation_seconds = time.perf_counter() - start
        if observer.latent is None:
            raise RuntimeError('Final continuous latent was not produced')
        token_audit, residual_audit = token.finalize(), residual.finalize()
        torch.cuda.empty_cache()
        with torch.amp.autocast('cuda', enabled=False):
            decoded = self.vae.decode(observer.latent.to(self.device))
        raw_model = ((decoded.detach().float().clamp(-1, 1) + 1.0) * 0.5).permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.float32)
        raw, valid = inverse_shift_with_source_fill(raw_model, source, dx, dy)
        output = restore_native(raw, geometry)
        audit = {'variant': variant, 'method': asdict(method), 'sampling': asdict(sampling),
                 'seed': seed, 'dataset': dataset, 'stratum': stratum, 'original_prompt': original_prompt,
                 'model_prompt': model_prompt, 'geometry': asdict(geometry), 'input': input_audit,
                 'route': route, 'phase_dx': int(dx), 'phase_dy': int(dy),
                 'phase_tier': phase.selected_tier if phase else 'disabled',
                 'mask_ratio': ratio, 'generation_seconds': generation_seconds,
                 'token': token_audit, 'residual': residual_audit}
        arrays = None
        if return_arrays:
            arrays = {**residual.export_arrays(), 'final_latent': observer.latent}
            for index, (sampled, mixed) in enumerate(zip(token.target_sampled_bits_cpu, token.mixed_bits_cpu)):
                arrays[f'scale_{index:02d}_sampled_bits'] = sampled
                arrays[f'scale_{index:02d}_mixed_bits'] = mixed
            for index, provenance in enumerate(token.provenance_maps_cpu):
                arrays[f'scale_{index:02d}_provenance'] = provenance
        return output, audit, arrays
