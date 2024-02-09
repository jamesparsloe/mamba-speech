from pydantic import BaseModel


class BaseConfig(BaseModel):
    n_text_tokens: int = 256

    max_duration: float = 16.0

    dac_model_name: str = "44khz"
    codebook_size: int = 1024
    n_quantizers: int = (
        9  # number of quantizers to model - must be <= codec.n_codebooks
    )

    d_model: int = 512
    n_layer: int = 12
    pad_vocab_size_multiple: int = 8
    dropout: float = 0.0

    @property
    def n_tokens(self):
        return self.n_text_tokens + self.codebook_size * self.n_quantizers

    @property
    def bos_token_id(self):
        return self.n_tokens

    @property
    def boa_token_id(self):
        return self.n_tokens + 1

    @property
    def eos_token_id(self):
        return self.n_tokens + 1 + 1

    @property
    def pad_token_id(self):
        return self.n_tokens + 1 + 1 + 1

    @property
    def vocab_size(self):
        return self.n_tokens + 1 + 1 + 1 + 1
