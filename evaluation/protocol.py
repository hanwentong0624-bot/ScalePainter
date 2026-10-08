"""Frozen v0.1 metric definitions, independent of generation environments."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

def _pairwise_pickscore_probability(text_features: torch.Tensor, image_features: torch.Tensor, logit_scale: torch.Tensor | float) -> torch.Tensor:
    """计算每个提示词在“生成图/源图”二选一中的生成图偏好概率。

    ``image_features`` 的形状必须为 ``[N, 2, D]``，第二维依次为生成图、
    源图。该函数只实现 PickScore 官方的成对 softmax 数学部分，便于脱离
    大模型做单元测试。
    """
    if text_features.ndim != 2 or image_features.ndim != 3:
        raise ValueError('PickScore 特征形状必须分别为 [N,D] 与 [N,2,D]')
    if image_features.shape[1] != 2:
        raise ValueError('PickScore 每个案例必须恰好包含生成图和源图')
    if text_features.shape[0] != image_features.shape[0]:
        raise ValueError('PickScore 文本与图像对的 batch 数不一致')
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = torch.einsum('nd,nkd->nk', text_features, image_features)
    logits = logits * torch.as_tensor(logit_scale, dtype=logits.dtype, device=logits.device)
    return torch.softmax(logits, dim=-1)[:, 0]

def _metric_prompts(case: dict[str, Any]) -> tuple[str, str]:
    """返回全图提示词和局部提示词。"""
    global_prompt = str(case.get('prompt_full') or case.get('global_prompt') or case.get('target_prompt') or case.get('caption'))
    local_prompt = str(case.get('prompt_mask_rich') or case.get('local_prompt') or case.get('target_prompt') or global_prompt)
    return (global_prompt, local_prompt)

def _masked_psnr(source: np.ndarray, output: np.ndarray, region: np.ndarray) -> float:
    """在指定区域计算 PSNR，输入为 0-1 RGB 数组。"""
    difference = source[region] - output[region]
    mse = float(np.mean(difference * difference))
    if mse <= 0.0:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))

def _masked_ssim_x10(source: np.ndarray, output: np.ndarray, region: np.ndarray) -> float:
    """在指定区域的包围框内计算 SSIM map 均值，并按 Token Painter 表格乘 10。"""
    ys, xs = np.where(region)
    if ys.size == 0 or xs.size == 0:
        return float('nan')
    y0, y1 = (int(ys.min()), int(ys.max()) + 1)
    x0, x1 = (int(xs.min()), int(xs.max()) + 1)
    source_crop = source[y0:y1, x0:x1]
    output_crop = output[y0:y1, x0:x1]
    region_crop = region[y0:y1, x0:x1]
    minimum = min(source_crop.shape[:2])
    window = min(7, minimum if minimum % 2 == 1 else minimum - 1)
    if window < 3:
        return float('nan')
    _, score_map = structural_similarity(source_crop, output_crop, channel_axis=2, data_range=1.0, win_size=window, full=True)
    if score_map.ndim == 3:
        score_map = score_map.mean(axis=2)
    return float(np.mean(score_map[region_crop]) * 10.0)

def _calculate_core_metrics(calculator: Any, row: dict[str, Any], metric_size: int) -> dict[str, Any]:
    """计算 Token Painter 八项中的六项基础指标和三项背景指标。"""
    source = _resize_image(Path(row['source_path']), metric_size)
    output = _resize_image(Path(row['raw_path']), metric_size)
    mask = _resize_mask(Path(row['mask_path']), metric_size)
    image_reward = calculator.calculate_image_reward(output, row['global_prompt'])
    hps_raw = calculator.calculate_hpsv21_score(output, row['global_prompt'])
    local_clip = calculator.calculate_clip_similarity(output, row['local_prompt'], mask)
    source_array = np.asarray(source, dtype=np.float32) / 255.0
    output_array = np.asarray(output, dtype=np.float32) / 255.0
    edit_mask = mask[:, :, 0] > 0.5
    background = ~edit_mask
    background_composite = output_array.copy()
    background_composite[edit_mask] = source_array[edit_mask]
    background_composite_image = Image.fromarray(np.clip(background_composite * 255.0, 0, 255).astype(np.uint8))
    bg_lpips = calculator.calculate_lpips(source, background_composite_image)
    bg_psnr = _masked_psnr(source_array, output_array, background)
    bg_ssim_x10 = _masked_ssim_x10(source_array, output_array, background)
    result = dict(row)
    result.update({'raw_all_black': bool(float(output_array.max()) == 0.0), 'ImageReward': float(np.asarray(image_reward).reshape(-1)[0]), 'IR_x10': float(np.asarray(image_reward).reshape(-1)[0]) * 10.0, 'HPSv2_raw': float(hps_raw), 'HPSv2_x100': float(hps_raw) * 100.0, 'CLIP_S_x100': float(local_clip), 'BG_PSNR': float(bg_psnr), 'BG_LPIPS': float(bg_lpips) if bg_lpips is not None else np.nan, 'BG_LPIPS_x1000': float(bg_lpips) * 1000.0 if bg_lpips is not None else np.nan, 'BG_SSIM_x10': float(bg_ssim_x10)})
    return result

def _resize_image(path: Path, size: int) -> Image.Image:
    """读取 RGB 图像并按 FreeInpaint 评测尺寸 resize。"""
    image = Image.open(path).convert('RGB')
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image

def _resize_mask(path: Path, size: int) -> np.ndarray:
    """读取二值编辑 mask，返回 HWC float 数组，1 表示编辑区域。"""
    mask = Image.open(path).convert('L')
    if mask.size != (size, size):
        mask = mask.resize((size, size), Image.Resampling.NEAREST)
    binary = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)
    return binary[:, :, np.newaxis]
