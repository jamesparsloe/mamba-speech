import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer


class MambaSpeechConfig(BaseModel):
    n_text_tokens: int = 256

    dac_model_name: str = "44khz"
    codebook_size: int = 1024
    n_quantizers: int = (
        9  # number of quantizers to model - must be <= codec.n_codebooks
    )

    # backbone/MambaLMHeadModel config
    d_model: int = 512
    n_layer: int = 12
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8


class MambaSpeech(nn.Module):
    def __init__(self, config: MambaSpeechConfig):
        super().__init__()
        self.config = config

        vocab_size = config.n_text_tokens + config.n_quantizers * config.codebook_size

        self.backbone = MambaLMHeadModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=vocab_size,
            rms_norm=config.rms_norm,
            residual_in_fp32=config.residual_in_fp32,
            fused_add_norm=config.fused_add_norm,
            pad_vocab_size_multiple=config.pad_vocab_size_multiple,
        )

    def forward(self, texts: list[str], waveforms: list[Tensor]):
        return self.model(input_ids)


if __name__ == "__main__":
    # model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # print(model.config)
    device = "cuda"
    config = MambaSpeechConfig(n_quantizers=3)
    model = MambaSpeech(config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

    print(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)

    print(tokenizer.special_tokens_map)

    # get token from id
    print(tokenizer.decode(2))

    text = "Hello, World!"

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(input_ids)

    print(tokenizer.batch_decode(input_ids))
