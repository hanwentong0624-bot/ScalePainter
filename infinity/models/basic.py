"""
Definitions of blocks of VAR transformer model.
"""
import math
import os
from functools import partial
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, drop_path
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from flash_attn import flash_attn_varlen_kvpacked_func
from torch.nn.functional import scaled_dot_product_attention as slow_attn
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False

    def rms_norm_impl(x, weight, epsilon):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon)) * weight

def precompute_rope2d_freqs_grid(dim, dynamic_resolution_h_w, rope2d_normalized_by_hw, pad_to_multiplier=1, max_height=2048 // 16, max_width=2048 // 16, base=10000.0, device=None, scaling_factor=1.0):
    half_dim = dim // 2
    inv_freq = 1.0 / base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim)
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
    t_height = t_height / scaling_factor
    freqs_height = torch.outer(t_height, inv_freq)
    t_width = t_width / scaling_factor
    freqs_width = torch.outer(t_width, inv_freq)
    freqs_grid_map = torch.concat([freqs_height[:, None, :].expand(-1, max_width, -1), freqs_width[None, :, :].expand(max_height, -1, -1)], dim=-1)
    freqs_grid_map = torch.stack([torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0)
    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]['1M']['scales']
        _, ph, pw = scale_schedule[-1]
        max_edge_length = freqs_grid_map.shape[1]
        if ph >= pw:
            uph, upw = (max_edge_length, int(max_edge_length / ph * pw))
        else:
            uph, upw = (int(max_edge_length / pw * ph), max_edge_length)
        rope_cache_list = []
        for _, ph, pw in scale_schedule:
            ph_mul_pw = ph * pw
            if rope2d_normalized_by_hw == 1:
                rope_cache = F.interpolate(freqs_grid_map[:, :uph, :upw, :].permute([0, 3, 1, 2]), size=(ph, pw), mode='bilinear', align_corners=True)
                rope_cache = rope_cache.permute([0, 2, 3, 1])
            elif rope2d_normalized_by_hw == 2:
                _, uph, upw = scale_schedule[-1]
                indices = torch.stack([(torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw), (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw)], dim=-1).round().int()
                indices = indices.reshape(-1, 2)
                rope_cache = freqs_grid_map[:, indices[:, 0], indices[:, 1], :]
                rope_cache = rope_cache.reshape(2, ph, pw, -1)
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :ph, :pw, :]
            else:
                raise ValueError(f'Unknown rope2d_normalized_by_hw: {rope2d_normalized_by_hw}')
            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))
        cat_rope_cache = torch.cat(rope_cache_list, 1)
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier, half_dim)
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)
        cat_rope_cache = cat_rope_cache[:, None, None, None]
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
            tmp_scale_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_scale_schedule))] = cat_rope_cache
    return rope2d_freqs_grid

def apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind):
    qk = torch.stack((q, k), dim=0)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope_key = str(tuple(scale_schedule))
        rope_grid = rope2d_freqs_grid[rope_key].to(qk.device)
        if torch.is_grad_enabled() and rope_grid.is_inference():
            rope_grid = rope_grid.clone()
        else:
            rope2d_freqs_grid[rope_key] = rope_grid
        assert start + seq_len <= rope_grid.shape[4]
        rope_cache = rope_grid[:, :, :, :, start:start + seq_len]
        qk = qk.reshape(*qk.shape[:-1], -1, 2)
        qk = torch.stack([rope_cache[0] * qk[..., 0] - rope_cache[1] * qk[..., 1], rope_cache[1] * qk[..., 0] + rope_cache[0] * qk[..., 1]], dim=-1)
        qk = qk.reshape(*qk.shape[:-2], -1)
        q, k = qk.unbind(dim=0)
    return (q, k)

class FastRMSNorm(nn.Module):

    def __init__(self, C, eps=1e-06, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))

    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)

    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'

def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()

class FFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0, fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = -1

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias, activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0, heuristic=self.heuristic, process_group=None))
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

class FFNSwiGLU(nn.Module):

    def __init__(self, in_features, hidden_features, out_features=None, drop=0.0, fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x))))

    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

class SelfAttention(nn.Module):

    def __init__(self, embed_dim=768, num_heads=12, proj_drop=0.0, tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        self.num_heads, self.head_dim = (num_heads, embed_dim // num_heads)
        self.tau, self.cos_attn = (tau, cos_attn)
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = (nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim)))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        self.caching = False
        self.cached_k = None
        self.cached_v = None
        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    def kv_caching(self, enable: bool):
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        B, L, C = x.shape
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash:
            q, k, v = qkv.unbind(dim=2)
            L_dim = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            L_dim = 2
        if self.cos_attn:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
            v = v.contiguous()
        else:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind)
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
        if self.using_flash:
            if attn_bias_or_two_vector is not None:
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
        elif self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'

class CrossAttention(nn.Module):

    def __init__(self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12, proj_drop=0.0, cos_attn=False):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = (num_heads, embed_dim // num_heads)
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim * 2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim)
        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        if self.cos_attn:
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        cu_seqlens_q = torch.arange(0, Lq * (B + 1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'

class SelfAttnBlock(nn.Module):

    def __init__(self, embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, tau=1, cos_attn=False, swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False):
        super(SelfAttnBlock, self).__init__()
        self.C, self.D = (embed_dim, cond_dim)
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.attn = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn, attn_fn=attn_fn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-06)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim ** 0.5)
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector):
        with torch.cuda.amp.autocast(enabled=False):
            if self.shared_aln:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        if self.fused_ada_norm is None:
            x = x + self.drop_path(self.attn(self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1), attn_bias_or_two_vector=attn_bias_or_two_vector).mul_(gamma1))
            x = x + self.drop_path(self.ffn(self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2)).mul(gamma2))
        else:
            x = x + self.drop_path(self.attn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1), attn_bias_or_two_vector=attn_bias_or_two_vector).mul_(gamma1))
            x = x + self.drop_path(self.ffn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2))
        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}'

class CrossAttnBlock(nn.Module):

    def __init__(self, embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0, tau=1, cos_attn=False, swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False, use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False):
        super(CrossAttnBlock, self).__init__()
        self.C, self.D = (embed_dim, cond_dim)
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sa = SelfAttention(embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn, use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw)
        self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-06)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim ** 0.5)
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        self.checkpointing_sa_only = checkpointing_sa_only

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0):
        with torch.cuda.amp.autocast(enabled=False):
            if self.shared_aln:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        if self.fused_norm_func is None:
            x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn(self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2)).mul(gamma2))
        else:
            x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind)
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2))
        return x

    def extra_repr(self) -> str:
        return f"shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={('<learnable>' if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma)}"

class AdaLNBeforeHead(nn.Module):

    def __init__(self, C, D, act: bool, norm_layer: partial, fused_norm_func=None):
        super().__init__()
        self.C, self.D = (C, D)
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-06)
        lin = nn.Linear(D, 2 * C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)

    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        else:
            return self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift)

def main():
    dev = 'cpu'
    rng = torch.Generator(device=dev)
    rng.manual_seed(0)
    B, H, cq, ckv = (4, 8, 64, 96)
    Cq = H * cq
    Ckv = H * ckv
    Li = [5, 4, 7, 6]
    Lq = 10
    L = max(Li)
    attn_bias = torch.zeros(B, 1, Lq, L, device=dev)
    for i, x in enumerate(Li):
        attn_bias[i, 0, :, x:] = -torch.inf
    q = torch.randn(B, Lq, H, cq, generator=rng, device=dev)
    k = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    v = torch.randn(B, L, H, ckv, generator=rng, device=dev)
    tq, tk, tv = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    seqlen_k = torch.tensor(Li, dtype=torch.int32, device=dev)
    cu_seqlens_k = F.pad(torch.cumsum(seqlen_k, dim=0, dtype=torch.torch.int32), (1, 0))
    kv = torch.stack([k, v], dim=2)
    kv_compact = torch.cat([kv[i, :Li[i]] for i in range(B)], dim=0)
    ca = CrossAttention(for_attn_pool=False, embed_dim=Cq, kv_dim=Ckv, num_heads=H)
    CrossAttention.forward
    ca(q, (kv_compact, cu_seqlens_k, max(Li))).mean().backward()
if __name__ == '__main__':
    main()
