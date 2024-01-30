from pydantic import BaseModel

from .model import MambaSpeechConfig


class TrainConfig(BaseModel):
    seed: int = 42

    # batch_size: int = 128
    batch_size: int = 16
    lr: float = 3e-4
    gradient_accumulation_steps: int = 4

    warmup_steps: int = 1_000
    steps: int = 100_000

    weight_decay: float = 0.1
    max_norm: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)

    val_items: int = 10
    val_every: int = 5_000
    checkpoint_every: int = 1_000
    log_every: int = 10


class Config(BaseModel):
    train: TrainConfig = TrainConfig()
    model: MambaSpeechConfig = MambaSpeechConfig()
