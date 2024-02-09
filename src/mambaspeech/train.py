import os
import time
from functools import partial
from typing import Literal

import click
import dac
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from einops import rearrange
from flash_attn.models.gpt import GPTLMHeadModel
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
from torchaudio.transforms import Resample
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import wandb

from .config import Config
from .constants import CACHE_DIR
from .model import MambaSpeech, MambaSpeechConfig
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


def tokenize(text: str, boa_token_id: int | None = None):
    token_ids = list(text.encode("utf-8"))
    if boa_token_id is not None:
        token_ids = token_ids + [boa_token_id]
    return torch.tensor(token_ids)


def collate(batch: list[tuple[str, Tensor]], *, boa_token_id: int):
    waveforms_lengths = []
    waveforms = []
    texts = []
    texts_lengths = []
    _texts = []

    for text, waveform in batch:
        _texts.append(text)
        waveforms_lengths.append(waveform.size(0))
        waveforms.append(waveform)
        tokenized = tokenize(text, boa_token_id=boa_token_id)
        texts.append(tokenized)
        texts_lengths.append(tokenized.size(-1))

    waveforms = pad_sequence(waveforms, batch_first=True)
    waveforms = rearrange(waveforms, "B T C -> B C T")

    waveforms_lengths = torch.tensor(waveforms_lengths)

    return {
        "_texts": _texts,
        "texts": texts,
        "texts_lengths": texts_lengths,
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
@click.option("--overfit", is_flag=True)
def main(config_path: str, edit: bool, overfit: bool):
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

    print(f"Saving checkpoints to {run_dir}")

    if train_config.dataset == "ljspeech":
        ds_sample_rate = 22050
        ds = torchaudio.datasets.LJSPEECH(CACHE_DIR, download=True)
        val_size = 100
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    elif train_config.dataset == "libritts":
        ds_sample_rate = 24000

        splits = []

        for url in ["train-clean-100", "train-clean-360", "train-other-500"]:
            root = os.path.join(CACHE_DIR, "libritts", url)
            os.makedirs(root, exist_ok=True)
            splits.append(torchaudio.datasets.LIBRITTS(root, url=url, download=True))

        ds = torch.utils.data.ConcatDataset(splits)
        val_size = 1000
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])

        print(f"{len(train_ds)=} {len(val_ds)=}")
    else:
        ds_sample_rate = 16000
        ds = torchaudio.datasets.SPEECHCOMMANDS(CACHE_DIR, download=True)

    dac_sample_rate = 44100
    resample = Resample(
        orig_freq=ds_sample_rate,
        new_freq=dac_sample_rate,
    ).to(device)

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).eval().to(device)

    n_text_tokens = model_config.n_text_tokens

    bos_token_id = model_config.bos_token_id
    boa_token_id = model_config.boa_token_id
    eos_token_id = model_config.eos_token_id
    pad_token_id = model_config.pad_token_id

    max_duration = model_config.max_duration

    _collate = partial(collate, boa_token_id=boa_token_id)

    dp = (
        to_iter_datapipe(train_ds)
        .map(lambda x: (x[3], rearrange(x[0], "C T -> T C")))
        .filter(lambda x: x[1].size(0) / ds_sample_rate < max_duration)
    )

    effective_batch_size = (
        train_config.batch_size * train_config.gradient_accumulation_steps
    )

    steps_per_epoch = len(train_ds) // effective_batch_size

    print(f"{steps_per_epoch=}")

    shuffle_buffer_size = 10 * effective_batch_size

    if overfit:
        shuffle_buffer_size = effective_batch_size
        dp = dp.header(effective_batch_size)

    dp = (
        dp.cycle()
        .shuffle(buffer_size=shuffle_buffer_size)
        .batch(train_config.batch_size, drop_last=True)
        .collate(_collate)
    )

    val_dp = (
        to_iter_datapipe(val_ds)
        .map(lambda x: (x[3], rearrange(x[0], "C T -> T C")))
        .filter(lambda x: x[1].size(0) / ds_sample_rate < max_duration)
        .batch(train_config.batch_size, drop_last=True)
        .collate(_collate)
    )

    num_workers = min(os.cpu_count() - 1, 4)
    rs = MultiProcessingReadingService(num_workers=num_workers)
    dl = DataLoader2(dp, reading_service=rs)

    val_dl = DataLoader2(val_dp, reading_service=rs)

    dl = iter(dl)

    step = 0

    amp_dtype = torch.bfloat16

    min_lr = train_config.lr / 10.0

    # fixed shape
    seqlen = None

    if model_config.kind == "gptspeech":
        seqlen = model_config.seqlen

        # https://github.com/Dao-AILab/flash-attention/tree/main/training#model-components
        gpt2_config = GPT2Config(
            vocab_size=model_config.vocab_size,
            n_positions=seqlen,
            n_embd=model_config.d_model,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            scale_attn_by_inverse_layer_idx=True,
            rotary_emb_fraction=0.5,
            use_flash_attn=True,
            fused_mlp=True,
            fused_bias_fc=True,
            fused_dropout_add_ln=True,
            pad_vocab_size_multiple=model_config.pad_vocab_size_multiple,
        )
        model = GPTLMHeadModel(gpt2_config).to(device)
    else:
        model = MambaSpeech(model_config).to(device)

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

    @torch.no_grad()
    def preprocess(
        texts: list[Tensor],
        waveforms: Tensor,
        waveforms_lengths: Tensor,
        seqlen: int | None = None,
    ) -> tuple[Tensor, Tensor]:
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

        audio_tokens = flatten_audio_tokens(audio_tokens)  # (B, n_quantizers * T)

        batched = []

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

            batched.append(token_ids)

        token_ids = pad_sequence(batched, batch_first=True, padding_value=pad_token_id)

        if seqlen is not None:
            T = token_ids.size(-1)
            assert T <= seqlen, f"token_ids size {T} > seqlen {seqlen}"
            pad = seqlen - token_ids.size(-1) + 1  # lose 1 for shift
            token_ids = F.pad(token_ids, (0, pad), value=pad_token_id)

        input_ids = token_ids[:, :-1].contiguous()
        target_ids = token_ids[:, 1:].contiguous()

        return input_ids, target_ids

    batch = next(dl)
    texts = batch["texts"]
    texts_lengths = batch["texts_lengths"]
    waveforms = batch["waveforms"].to(device, non_blocking=True)
    waveforms_lengths = batch["waveforms_lengths"].to(device, non_blocking=True)

    if overfit:
        print(f"Overfitting on the following samples:")
        for text in batch["_texts"]:
            print(f"\t{text}")

        print()

    t1 = time.perf_counter()

    while step < train_config.steps:
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for micro_step in range(train_config.gradient_accumulation_steps):
            input_ids, target_ids = preprocess(
                texts, waveforms, waveforms_lengths, seqlen=seqlen
            )

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
            texts_lengths = batch["texts_lengths"]
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
            samples = effective_batch_size * train_config.log_every
            throughput = samples / (t2 - t1)

            metrics = {
                "train/loss": loss.item() * train_config.gradient_accumulation_steps,
                "train/grad_norm": grad_norm.item(),
                "train/throughput": throughput,
                "train/lr": lr,
            }

            if os.environ.get("WANDB_MODE") == "disabled":
                print(f"{step=} {metrics}")

            wandb.log(
                metrics,
                step=step,
            )
            t1 = t2

        if step % train_config.val_every == 0:
            model.eval()
            val_loss_total = 0.0
            val_size = 0
            for val_batch in val_dl:
                val_texts = val_batch["texts"]
                val_texts_lengths = val_batch["texts_lengths"]
                val_waveforms = val_batch["waveforms"].to(device, non_blocking=True)
                val_waveforms_lengths = val_batch["waveforms_lengths"].to(
                    device, non_blocking=True
                )

                val_input_ids, val_target_ids = preprocess(
                    val_texts, val_waveforms, val_waveforms_lengths, seqlen=seqlen
                )

                with torch.no_grad():
                    val_output = model(val_input_ids)
                    val_logits = val_output.logits

                    val_loss = F.cross_entropy(
                        val_logits.view(-1, val_logits.size(-1)),
                        val_target_ids.view(-1),
                        ignore_index=pad_token_id,
                    )

                val_loss_total += train_config.batch_size * val_loss.item()
                val_size += train_config.batch_size

            val_loss = val_loss_total / val_size

            wandb.log(
                {
                    "val/loss": val_loss,
                },
                step=step,
            )
            model.train()

        if step % train_config.checkpoint_every == 0:
            checkpoint = {
                "step": step,
                "config": config.model_dump(),
                "model": model.state_dict(),
            }

            checkpoint_dir = os.path.join(run_dir, f"{step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            path = os.path.join(checkpoint_dir, "model.pt")
            torch.save(checkpoint, path)

            print(f"Checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
