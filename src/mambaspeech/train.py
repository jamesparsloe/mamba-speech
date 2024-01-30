import os
import time
from functools import partial

import click
import dac
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from einops import rearrange
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torchaudio.transforms import Resample
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

import wandb

from .config import Config
from .constants import CACHE_DIR
from .utils import seed_all, to_iter_datapipe, warmup_then_cosine_decay


def configure_optimizers(self, *, weight_decay, lr: float, betas):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

    return optimizer


def tokenize(text: str):
    return torch.tensor(list(text.encode("utf-8")))


def collate(batch: list[tuple[str, Tensor]], *, bos_token_id: int):
    waveforms_lengths = []
    waveforms = []
    texts = []

    for text, waveform in batch:
        waveforms_lengths.append(waveform.size(0))
        waveforms.append(waveform)
        texts.append(tokenize(text))

    waveforms = pad_sequence(waveforms, batch_first=True)
    waveforms = rearrange(waveforms, "B T C -> B C T")

    waveforms_lengths = torch.tensor(waveforms_lengths)

    return {
        "texts": texts,
        "waveforms": waveforms,
        "waveforms_lengths": waveforms_lengths,
    }


def flatten_audio_tokens(audio_tokens: Tensor):
    audio_tokens = rearrange(audio_tokens, "B Q T -> B T Q")
    audio_tokens = rearrange(audio_tokens, "B T Q -> B (T Q)")
    return audio_tokens


def unflatten_audio_tokens(audio_tokens: Tensor, T: int):
    audio_tokens = rearrange(audio_tokens, "B (T Q) -> B T Q", T=T)
    audio_tokens = rearrange(audio_tokens, "B T Q -> B Q T")
    return audio_tokens


def set_eos_id(t: Tensor, eos_id: int, pad_id: int):
    eos_indices = ((t == pad_id).cumsum(dim=-1) == 0).sum(dim=-1, keepdim=True).long()

    batch_range = torch.arange(t.shape[0], device=t.device, dtype=torch.int64)
    batch_range = rearrange(batch_range, "... -> ... 1")

    t = F.pad(t, (0, 1), value=pad_id)
    t[batch_range, eos_indices] = eos_id

    return t


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
    # ds_sample_rate = 16000
    # ds = torchaudio.datasets.SPEECHCOMMANDS(CACHE_DIR, download=True)

    ds_sample_rate = 22050
    ds = torchaudio.datasets.LJSPEECH(CACHE_DIR, download=True)

    # ds_sample_rate = 24000

    # ds = []

    # for url in ["train-clean-100", "train-clean-360", "train-other-500"]:
    #     root = os.path.join(CACHE_DIR, "libritts", url)
    #     os.makedirs(root, exist_ok=True)
    #     ds.append(torchaudio.datasets.LIBRITTS(root, url=url, download=True))

    # ds = torch.utils.data.ConcatDataset(ds)

    dac_sample_rate = 44100
    resample = Resample(
        orig_freq=ds_sample_rate,
        new_freq=dac_sample_rate,
    ).to(device)

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).eval().to(device)

    max_duration = 8.0

    n_text_tokens = 256

    n_tokens = n_text_tokens + model_config.codebook_size * model_config.n_quantizers
    bos_token_id = n_tokens
    eos_token_id = n_tokens + 1
    pad_token_id = n_tokens + 1 + 1
    vocab_size = n_tokens + 1 + 1 + 1

    _collate = partial(collate, bos_token_id=bos_token_id)

    dp = (
        to_iter_datapipe(ds)
        .map(lambda x: (x[3], rearrange(x[0], "C T -> T C")))
        .filter(lambda x: x[1].size(0) / ds_sample_rate < max_duration)
        .cycle()
        .shuffle(buffer_size=10 * train_config.batch_size)
        .batch(train_config.batch_size, drop_last=True)
        .collate(_collate)
    )

    num_workers = os.cpu_count() - 1
    rs = MultiProcessingReadingService(num_workers=num_workers)
    dl = DataLoader2(dp, reading_service=rs)

    dl = iter(dl)

    step = 0

    amp_dtype = torch.bfloat16

    mamba_config = MambaConfig(
        # d_model=768,
        # n_layer=24,
        d_model=512,
        n_layer=12,
        vocab_size=vocab_size,
        ssm_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
    )
    min_lr = train_config.lr / 10.0

    model = MambaLMHeadModel(mamba_config).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{n_parameters / 1e6:.1f}M parameters")

    optimizer = configure_optimizers(
        model,
        lr=min_lr,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
    )

    quantizer_offsets = n_text_tokens + model_config.codebook_size * torch.arange(
        model_config.n_quantizers, device=device
    )
    quantizer_offsets = rearrange(quantizer_offsets, "Q -> Q 1")

    get_lr = partial(
        warmup_then_cosine_decay,
        warmup_steps=train_config.warmup_steps,
        steps=train_config.steps,
        min_lr=min_lr,
        max_lr=train_config.lr,
    )

    batch = next(dl)
    texts = batch["texts"]
    waveforms = batch["waveforms"].to(device, non_blocking=True)
    waveforms_lengths = batch["waveforms_lengths"].to(device, non_blocking=True)

    t1 = time.perf_counter()

    while step < train_config.steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(train_config.gradient_accumulation_steps):
            with torch.no_grad():
                length_before = waveforms.size(-1)
                waveforms = resample(waveforms)
                resample_factor = waveforms.size(-1) / length_before
                waveforms = codec.preprocess(waveforms, dac_sample_rate)
                _, audio_tokens, _, _, _ = codec.encode(waveforms)

                audio_tokens_lengths = (
                    (resample_factor * waveforms_lengths / codec.hop_length)
                    .ceil()
                    .to(torch.int32)
                )

                B, Q, T = audio_tokens.size()

                audio_tokens = audio_tokens[:, : model_config.n_quantizers]
                audio_tokens = audio_tokens + quantizer_offsets

                audio_tokens = flatten_audio_tokens(
                    audio_tokens
                )  # (B, n_quantizers * T)

                token_ids_batch = []

                for b in range(B):
                    length = model_config.n_quantizers * audio_tokens_lengths[b]

                    token_ids = torch.cat(
                        (
                            texts[b].to(device),
                            audio_tokens[b, :length],
                        ),
                    )
                    token_ids = F.pad(token_ids, (1, 0), value=bos_token_id)
                    token_ids = F.pad(token_ids, (0, 1), value=eos_token_id)

                    token_ids_batch.append(token_ids)

                token_ids = pad_sequence(
                    token_ids_batch, batch_first=True, padding_value=pad_token_id
                )

                input_ids = token_ids[:, :-1].contiguous()
                target_ids = token_ids[:, 1:].contiguous()

            with torch.amp.autocast(dtype=amp_dtype, device_type="cuda", enabled=True):
                output = model(input_ids)
                logits = output.logits

                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=pad_token_id,
                )

                loss = loss / train_config.gradient_accumulation_steps

            batch = next(dl)
            texts = batch["texts"]
            waveforms = batch["waveforms"].to(device, non_blocking=True)
            waveforms_lengths = batch["waveforms_lengths"].to(device, non_blocking=True)

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
