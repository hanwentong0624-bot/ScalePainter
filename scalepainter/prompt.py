"""ScalePainter prompt. Derived from the verified research implementation."""
from __future__ import annotations
from dataclasses import dataclass
E125_VARIANT = 'e125_g20_masksize_only'
_SIZE_WORDS = {'small': 'small', 'medium': 'medium-sized', 'large': 'large'}

@dataclass(frozen=True)
class MaskSizePrompt:
    """保存一次尺寸提示适配及其可审计字段。"""
    variant: str
    original_prompt: str
    expanded_prompt: str
    mask_stratum: str
    size_word: str

    def to_dict(self) -> dict[str, str | bool]:
        """返回可直接写入 JSON 的审计记录。"""
        return {'variant': self.variant, 'original_prompt': self.original_prompt, 'expanded_prompt': self.expanded_prompt, 'mask_stratum': self.mask_stratum, 'size_word': self.size_word, 'complete_visible_phrase_added': False, 'external_language_model_used': False, 'source_or_target_image_read': False}

def add_mask_size_word(prompt: str, mask_stratum: str) -> MaskSizePrompt:
    """按冻结分层在原目标词前加入一个尺寸词。

    这里故意不做名词单复数、冠词或语义角色分析。OABench 提示本身是短目标
    名词，直接前置形容词既能保持实现简洁，也避免引入新的语言解析模块。
    """
    clean_prompt = ' '.join(str(prompt).strip().split())
    if not clean_prompt:
        raise ValueError('E125 原始目标提示不能为空')
    clean_stratum = str(mask_stratum).strip().lower()
    if clean_stratum not in _SIZE_WORDS:
        raise ValueError(f'E125 未知 mask stratum：{mask_stratum}')
    size_word = _SIZE_WORDS[clean_stratum]
    return MaskSizePrompt(variant=E125_VARIANT, original_prompt=clean_prompt, expanded_prompt=f'{size_word} {clean_prompt}', mask_stratum=clean_stratum, size_word=size_word)
