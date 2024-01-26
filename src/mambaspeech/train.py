import os
import time
from functools import partial

import click
import dac
import torch
import torch.nn.functional as F
import torchaudio
import wandb
import yaml
from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torchaudio.transforms import Resample
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

from .config import Config
from .constants import CACHE_DIR
from .utils import seed_all, to_iter_datapipe, warmup_then_cosine_decay


def collate(waveforms: list[Tensor]):
    waveforms = pad_sequence(waveforms, batch_first=True)
    waveforms = rearrange(waveforms, "B T C -> B C T")
    return waveforms


def flatten_audio_tokens(audio_tokens: Tensor):
    audio_tokens = rearrange(audio_tokens, "B Q T -> B T Q")
    audio_tokens = rearrange(audio_tokens, "B T Q -> B (T Q)")
    return audio_tokens


def unflatten_audio_tokens(audio_tokens: Tensor, T: int):
    audio_tokens = rearrange(audio_tokens, "B (T Q) -> B T Q", T=T)
    audio_tokens = rearrange(audio_tokens, "B T Q -> B Q T")
    return audio_tokens


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--edit", is_flag=True)
def main(config_path: str, edit: bool):
    assert os.getenv("WANDB_API_KEY"), "Please set WANDB_API_KEY"
    assert os.getenv("HF_TOKEN"), "Please set HF_TOKEN"

    name = "mambaspeech"

    device = "cuda"

    with open(config_path) as f:
        s = f.read()
        if edit:
            s = click.edit(s)
        config = Config(**yaml.safe_load(s))

    train_config = config.train
    model_config = config.model

    seed = train_config.seed

    seed_all(seed)

    run = wandb.init(project=name, config=config.model_dump())

    run_dir = os.path.join("./runs", run.id)
    os.makedirs(run_dir, exist_ok=True)

    # NOTE do this next
    # ds = torchaudio.datasets.LJSPEECH(CACHE_DIR, download=True)
    ds_sample_rate = 16000
    dac_sample_rate = 44100
    resample = Resample(
        orig_freq=ds_sample_rate,
        new_freq=dac_sample_rate,
    ).to(device)

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).eval().to(device)

    ds = torchaudio.datasets.SPEECHCOMMANDS(CACHE_DIR, download=True)
    dp = (
        to_iter_datapipe(ds)
        .map(lambda x: rearrange(x[0], "C T -> T C"))
        .shuffle(buffer_size=100)
        .batch(train_config.batch_size, drop_last=True)
        .collate(collate)
    )

    num_workers = os.cpu_count() - 1
    rs = MultiProcessingReadingService(num_workers=num_workers)
    dl = DataLoader2(dp, reading_service=rs)

    dl = iter(dl)

    step = 0

    amp_dtype = torch.bfloat16

    n_tokens = model_config.codebook_size * model_config.n_quantizers
    bos_token_id = n_tokens
    eos_token_id = n_tokens + 1
    pad_token_id = n_tokens + 1 + 1
    vocab_size = n_tokens + 1 + 1 + 1

    mamba_config = MambaConfig(
        d_model=768,
        n_layer=24,
        vocab_size=vocab_size,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
    )
    min_lr = train_config.lr / 10.0

    get_lr = partial(
        warmup_then_cosine_decay,
        warmup_steps=train_config.warmup_steps,
        steps=train_config.steps,
        min_lr=min_lr,
        max_lr=train_config.lr,
    )

    model = MambaLMHeadModel(mamba_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=min_lr,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
    )

    quantizer_offsets = model_config.codebook_size * torch.arange(
        model_config.n_quantizers, device=device
    )
    quantizer_offsets = rearrange(quantizer_offsets, "Q -> Q 1")

    t1 = time.perf_counter()

    while step < train_config.steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        waveforms = next(dl)
        waveforms = waveforms.to(device, non_blocking=True)

        with torch.no_grad():
            waveforms = resample(waveforms)
            waveforms = codec.preprocess(waveforms, dac_sample_rate)
            _, audio_tokens, _, _, _ = codec.encode(waveforms)

            audio_tokens = audio_tokens[:, : model_config.n_quantizers]
            audio_tokens = audio_tokens + quantizer_offsets

            B, Q, T = audio_tokens.size()
            audio_tokens = flatten_audio_tokens(audio_tokens)
            input_ids = F.pad(audio_tokens, (1, 0), value=bos_token_id)
            target_ids = F.pad(audio_tokens, (0, 1), value=eos_token_id)

        with torch.amp.autocast(dtype=amp_dtype, device_type="cuda", enabled=True):
            output = model(input_ids)
            logits = output.logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=pad_token_id,
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_norm
        )

        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step % train_config.log_every == 0:
            t2 = time.perf_counter()
            samples = (
                train_config.batch_size
                * train_config.gradient_accumulation_steps
                * train_config.log_every
            )
            throughput = samples / (t2 - t1)

            wandb.log(
                {
                    "train/loss": loss.item()
                    * train_config.gradient_accumulation_steps,
                    "train/grad_norm": grad_norm.item(),
                    "train/throughput": throughput,
                    "train/lr": lr,
                },
                step=step,
            )
            t1 = t2

        if step % train_config.checkpoint_every == 0:
            checkpoint_dir = os.path.join(run_dir, f"{step:06d}")
            model.save_pretrained(checkpoint_dir)
            print(f"Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
