"""Load the released Infinity 8B sharded model without modifying checkpoints."""
from pathlib import Path

import torch
import torch.nn.functional as F


def encode_prompt(tokenizer, encoder, prompt):
    tokens = tokenizer(text=[prompt], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    features = encoder(input_ids=ids, attention_mask=mask)['last_hidden_state'].float()
    lengths = mask.sum(dim=-1).tolist()
    cumulative = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    compact = torch.cat([feat[:length] for length, feat in zip(lengths, features.unbind(0))], dim=0)
    return compact, lengths, cumulative, max(lengths)


def load_models(model_path, vae_path, text_encoder_path, device='cuda:0'):
    from transformers import AutoTokenizer, T5EncoderModel
    from transformers.modeling_utils import load_sharded_checkpoint
    from infinity.models.infinity import Infinity
    from infinity.models.bsq_vae.vae import vae_model

    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Infinity 8B inference requires CUDA and FlashAttention; CPU utilities remain available')
    torch.cuda.set_device(device)
    for value in [model_path, vae_path, text_encoder_path]:
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if not (Path(model_path) / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError('Expected the released four-shard safetensors directory')
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, revision=None, legacy=True, local_files_only=True)
    tokenizer.model_max_length = 512
    encoder = T5EncoderModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16, local_files_only=True).to(device)
    encoder.eval().requires_grad_(False)
    vae = vae_model(str(vae_path), 'dynamic', 14, 2**14, patch_size=8,
                    encoder_ch_mult=[1, 2, 4, 4], decoder_ch_mult=[1, 2, 4, 4], test_mode=True).to(device)
    vae.eval().requires_grad_(False)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        model = Infinity(
            vae_local=vae, text_channels=2048, text_maxlen=512,
            shared_aln=True, raw_scale_schedule=None, checkpointing='full-block',
            customized_flash_attn=False, fused_norm=True, pad_to_multiplier=128,
            use_flex_attn=False, add_lvl_embeding_only_first_block=1,
            use_bit_label=1, rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2,
            pn='1M', apply_spatial_patchify=1, inference_mode=True,
            train_h_div_w_list=[1.0], depth=40, embed_dim=3584,
            num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8,
        )
        # Preserve final dtypes without a full-float32 GPU loading peak.
        for block in model.unregistered_blocks:
            block.bfloat16()
        model.to(device)
        model.eval().requires_grad_(False)
        torch.cuda.empty_cache()
        incompatible = load_sharded_checkpoint(model, str(model_path), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f'Checkpoint key mismatch: {incompatible}')
        model.rng = torch.Generator(device=device)
    return model, vae, tokenizer, encoder
