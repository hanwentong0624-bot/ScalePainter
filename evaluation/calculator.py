"""Explicit local metric assets; no dependency on a research checkout."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


class CoreCalculator:
    def __init__(self, assets, device='cuda:0'):
        from transformers import BertTokenizer
        from torchmetrics.multimodal import CLIPScore
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        import ImageReward as RM
        from hpsv2 import img_score

        self.device = torch.device(device)
        self.clip_metric_calculator = CLIPScore(model_name_or_path=assets['clip']).to(self.device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(self.device)
        reward_dir = Path(assets['image_reward'])
        original = BertTokenizer.from_pretrained

        def local_tokenizer(name, *args, **kwargs):
            if name == 'bert-base-uncased':
                name = assets['bert_tokenizer']
                kwargs['local_files_only'] = True
            return original(name, *args, **kwargs)

        # ImageReward 1.5 hardcodes the tokenizer name in its constructor.
        # Scope the redirection to construction and restore the dependency method.
        with patch.object(BertTokenizer, 'from_pretrained', side_effect=local_tokenizer):
            self.imagereward_model = RM.load(str(reward_dir/'ImageReward.pt'), device=self.device,
                                             med_config=str(reward_dir/'med_config.json'))
        self.hps_model, _, self.hps_preprocess = img_score.create_model_and_transforms(
            'ViT-H-14', None, precision='amp', device=self.device, jit=False,
            force_quick_gelu=False, force_custom_text=False, force_patch_dropout=False,
            force_image_size=None, pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False,
        )
        checkpoint = torch.load(assets['hps'], map_location=self.device)
        self.hps_model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
        self.hps_model.to(self.device).eval()
        self.hps_tokenizer = img_score.get_tokenizer('ViT-H-14')

    def calculate_image_reward(self, image, prompt):
        return self.imagereward_model.score(prompt, [image])

    @torch.no_grad()
    def calculate_hpsv21_score(self, image, prompt):
        tensor = self.hps_preprocess(image).unsqueeze(0).to(self.device, non_blocking=True)
        text = self.hps_tokenizer([prompt]).to(self.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            output = self.hps_model(tensor, text)
            logits = output['image_features'] @ output['text_features'].T
            value = torch.diagonal(logits).cpu().numpy()[0]
        return value.item()

    def calculate_clip_similarity(self, image, prompt, mask=None):
        array = np.array(image)
        if mask is not None:
            binary = np.asarray(mask)
            if binary.ndim == 3:
                binary = binary.sum(axis=2)
            ys, xs = np.where(binary > 0)
            if len(ys):
                array = array[ys.min():ys.max()+1, xs.min():xs.max()+1]
        tensor = torch.tensor(array).permute(2, 0, 1).to(self.device)
        return self.clip_metric_calculator(tensor, prompt).cpu().item()

    def calculate_lpips(self, source, prediction, mask=None):
        if mask is not None:
            raise ValueError('The v0.1 evaluator supplies a background composite, not a cropped LPIPS mask')
        if source.size != prediction.size:
            source = source.resize(prediction.size)
        predicted = np.array(prediction).astype(np.float32) / 255
        expected = np.array(source).astype(np.float32) / 255
        predicted = torch.tensor(predicted).permute(2, 0, 1).unsqueeze(0).to(self.device)
        expected = torch.tensor(expected).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return self.lpips_metric_calculator(predicted*2-1, expected*2-1).cpu().item()
