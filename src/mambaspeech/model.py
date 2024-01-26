import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer


class MambaSpeechConfig(BaseModel):
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    base_namename: str = "state-spaces/mamba-130m"

    dac_model_name: str = "44khz"
    codebook_size: int = 1024
    n_quantizers: int = (
        9  # number of quantizers to model - must be <= codec.n_codebooks
    )


# TODO
class MambaTTS(nn.Module):
    def __init__(self, config: MambaSpeechConfig):
        super().__init__()
        self.config = config

        self.model = MambaLMHeadModel.from_pretrained(config.model_name)

        embedding = self.model.backbone.embedding
        n_tokens, embedding_dim = embedding.weight.size()

        n_audio_tokens = config.n_quantizers * config.codebook_size
        self.audio_eos_id = n_tokens + n_audio_tokens
        self.audio_pad_id = n_tokens + n_audio_tokens + 1
        n_audio_tokens = n_audio_tokens + 1 + 1

        n_new_tokens = n_tokens + n_audio_tokens

        new_embedding = nn.Embedding(n_new_tokens, embedding_dim)

        new_embedding.weight.data[:n_tokens] = embedding.weight.data[:n_tokens]
        self.model.backbone.embedding = new_embedding

        lm_head = self.model.lm_head
        n_tokens, in_features = lm_head.weight.size()
        bias = lm_head.bias
        new_lm_head = nn.Linear(in_features, n_new_tokens, bias=bias)
        new_lm_head.weight.data[:n_tokens] = lm_head.weight.data[:n_tokens]

        self.model.lm_head = new_lm_head

    def forward(self, input_ids: Tensor): ...


if __name__ == "__main__":
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    print(model.config)
