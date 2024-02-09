from typing import Literal
from .base import BaseConfig


class GPTSpeechConfig(BaseConfig):
    kind: Literal["gptspeech"]
    seqlen: int = 4096
    rotary_emb_fraction: float = 0.5
    head_dim: int = 64

    @property
    def n_head(self):
        return self.d_model // self.head_dim
