import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer


class MambaSpeechConfig(BaseModel):
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    base_name: str = "state-spaces/mamba-130m"

    dac_model_name: str = "44khz"
    codebook_size: int = 1024
    n_quantizers: int = (
        9  # number of quantizers to model - must be <= codec.n_codebooks
    )


class MambaTTS(nn.Module):
    def __init__(self, config: MambaSpeechConfig):
        super().__init__()
        self.config = config

        self.model = MambaLMHeadModel.from_pretrained(config.base_name)

        # copy across pretraind embeddings etc
        mamba_config = self.model.config

        embedding = self.model.backbone.embedding
        n_text_tokens, embedding_dim = embedding.weight.size()

        self.n_text_tokens = n_text_tokens

        n_audio_tokens = config.n_quantizers * config.codebook_size
        self.audio_bos_id = n_text_tokens + n_audio_tokens
        self.audio_eos_id = n_text_tokens + n_audio_tokens + 1
        self.audio_pad_id = n_text_tokens + n_audio_tokens + 1 + 1
        n_audio_tokens = n_audio_tokens + 1 + 1 + 1

        vocab_size = n_text_tokens + n_audio_tokens

        if vocab_size % mamba_config.pad_vocab_size_multiple != 0:
            vocab_size += mamba_config.pad_vocab_size_multiple - (
                vocab_size % mamba_config.pad_vocab_size_multiple
            )

        new_embedding = nn.Embedding(vocab_size, embedding_dim)

        new_embedding.weight.data[:n_text_tokens] = embedding.weight.data[
            :n_text_tokens
        ]
        self.model.backbone.embedding = new_embedding

        lm_head = self.model.lm_head
        n_text_tokens, in_features = lm_head.weight.size()
        bias = lm_head.bias
        new_lm_head = nn.Linear(in_features, vocab_size, bias=bias)
        new_lm_head.weight.data[:n_text_tokens] = lm_head.weight.data[:n_text_tokens]

        self.model.lm_head = new_lm_head

        self.model.tie_weights()

    def forward(self, input_ids: Tensor):
        return self.model(input_ids)


if __name__ == "__main__":
    # model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # print(model.config)
    device = "cuda"
    config = MambaSpeechConfig(n_quantizers=3)
    model = MambaTTS(config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)

    print(tokenizer.special_tokens_map)

    # get token from id
    print(tokenizer.decode(2))

    text = "Hello, World!"

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(input_ids)

    print(tokenizer.batch_decode(input_ids))
