"""Local-checkpoint preference scorers."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoProcessor, CLIPVisionModelWithProjection

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

class PickScoreScorer:
    """按 PickScore 官方定义计算生成图相对源图的成对偏好概率。"""

    def __init__(self, model_path: Path, processor_path: Path, device: str) -> None:
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(processor_path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_path), local_files_only=True).eval().to(self.device)

    @torch.no_grad()
    def score_pairs(self, generated_images: Sequence[Image.Image], source_images: Sequence[Image.Image], prompts: Sequence[str]) -> np.ndarray:
        """返回生成图相对源图的 PickScore 概率，并按论文表格乘以 100。"""
        if not len(generated_images) == len(source_images) == len(prompts):
            raise ValueError('PickScore 的生成图、源图与文本数量必须一致')
        paired_images = [image for pair in zip(generated_images, source_images) for image in pair]
        image_inputs = self.processor(images=paired_images, padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        text_inputs = self.processor(text=list(prompts), padding=True, truncation=True, max_length=77, return_tensors='pt').to(self.device)
        image_features = self.model.get_image_features(**image_inputs)
        text_features = self.model.get_text_features(**text_inputs)
        image_features = image_features.reshape(len(prompts), 2, -1)
        logit_scale = self.model.logit_scale.exp()
        probabilities = _pairwise_pickscore_probability(text_features, image_features, logit_scale)
        return (probabilities * 100.0).float().cpu().numpy()

class AestheticScorer:
    """复现 improved-aesthetic-predictor 的 CLIP-L/14 线性头。"""

    def __init__(self, clip_path: Path, head_path: Path, device: str) -> None:
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(clip_path), local_files_only=True)
        self.clip = CLIPVisionModelWithProjection.from_pretrained(str(clip_path), local_files_only=True).eval().to(self.device)
        projection_dim = int(self.clip.config.projection_dim)
        self.head = nn.Sequential(nn.Linear(projection_dim, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2), nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
        raw_state = torch.load(head_path, map_location='cpu', weights_only=True)
        state = {str(key).replace('layers.', ''): value for key, value in raw_state.items()}
        self.head.load_state_dict(state, strict=True)
        self.head.eval().to(self.device)

    @torch.no_grad()
    def score(self, images: Sequence[Image.Image]) -> np.ndarray:
        """返回美学预测器原始分数。"""
        inputs = self.processor(images=list(images), return_tensors='pt').to(self.device)
        features = self.clip(**inputs).image_embeds
        features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features).squeeze(-1).float().cpu().numpy()
