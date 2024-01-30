from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from mamba_ssm.utils.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
    InferenceParams,
    modify_logits_for_top_p_filtering,
    modify_logit_for_repetition_penalty,
    update_graph_cache,
)
import json
import torch
from torch import Tensor
from einops import rearrange
import dac
import torchaudio
import time
import gradio as gr
from mambaspeech.train import flatten_audio_tokens, tokenize
import torch.nn.functional as F


def unflatten_audio_tokens(audio_tokens: Tensor, T: int):
    audio_tokens = rearrange(audio_tokens, "B (T Q) -> B T Q", T=T)
    audio_tokens = rearrange(audio_tokens, "B T Q -> B Q T")
    return audio_tokens


def sample(logits, top_k=1, top_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1),
            ]
        else:
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(
                torch.softmax(logits_top, dim=-1), num_samples=1
            ).squeeze(dim=-1)


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    *,
    n_quantizers: int,
    text_offset: int,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """

    T_text = input_ids.size(-1)

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = 0

    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(
            max_seqlen=max_length, max_batch_size=batch_size
        )

    def get_logits(input_ids, inference_params):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)

        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        token = sample(logits, top_k=top_k, top_p=top_p, temperature=temperature)

        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        start.record()
    scores, sequences = [], [input_ids]
    sequences_cat = input_ids

    step = 0

    while not should_stop(sequences[-1], inference_params):
        logits = get_logits(sequences[-1], inference_params)

        rem = step % n_quantizers

        # mask = torch.full_like(logits, -float("inf"), device=device)
        ids = torch.arange(logits.size(-1), device=device)
        mask = torch.where(
            (ids >= rem * codebook_size + text_offset)
            & (ids < (rem + 1) * codebook_size + text_offset),
            0.0,
            -float("inf"),
        )
        logits = logits + mask

        scores.append(logits)
        inference_params.seqlen_offset += sequences[-1].shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1], inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)

        step += 1

        sequences.append(sampled_tokens)

    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")

    output_cls = (
        GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    )

    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


device = "cuda"


dac_path = dac.utils.download(model_type="44khz")
codec = dac.DAC.load(dac_path).eval().to(device)

# config_path = "runs/07gd8g19/002000/config.json"
# checkpoint_path = "runs/07gd8g19/002000/pytorch_model.bin"

# SpeechCommands
config_path = "runs/t8estdri/004000/config.json"
checkpoint_path = "runs/t8estdri/004000/pytorch_model.bin"

# LJSPEECH
name = "ljspeech"
config_path = "./runs/h9sz5600/012000/config.json"
checkpoint_path = "./runs/h9sz5600/012000/pytorch_model.bin"

name = "libritts"
config_path = "./runs/gwa4na10/034000/config.json"
checkpoint_path = "./runs/gwa4na10/034000/pytorch_model.bin"

text_condtioned = True
text_offset = 256
config_path = "./runs/egb5ural/002000/config.json"
checkpoint_path = "./runs/egb5ural/002000/pytorch_model.bin"

text_condtioned = True
text_offset = 256
config_path = "./runs/o9rzoxpa/004000/config.json"
checkpoint_path = "./runs/o9rzoxpa/004000/pytorch_model.bin"

text_condtioned = True
text_offset = 256
config_path = "./runs/mdghb8lx/000500/config.json"
checkpoint_path = "./runs/mdghb8lx/000500/pytorch_model.bin"

text_condtioned = True
text_offset = 256
config_path = "./runs/lzu5rp0v/002000/config.json"
checkpoint_path = "./runs/lzu5rp0v/002000/pytorch_model.bin"

with open(config_path, "r") as f:
    config = MambaConfig(**json.load(f))

state_dict = torch.load(checkpoint_path)

model = MambaLMHeadModel(config)
_ = model.load_state_dict(state_dict)
model = model.to(device).eval()

codebook_size = 1024
n_quantizers = 3
n_tokens = codebook_size * n_quantizers
bos_token_id = n_tokens


quantizer_offsets = text_offset + codebook_size * torch.arange(
    n_quantizers, device=device
)
quantizer_offsets = rearrange(quantizer_offsets, "Q -> Q 1")


def generate(text: str, temperature: float, top_k: int):
    id = int(time.time())

    # prompt_waveform, prompt_sample_rate = torchaudio.load(prompt_path)
    # prompt_waveform = torchaudio.functional.resample(
    #     prompt_waveform, prompt_sample_rate, codec.sample_rate
    # )

    # prompt_waveform = prompt_waveform.to(device)
    # prompt_waveform = rearrange(prompt_waveform, "C T -> 1 C T")
    # prompt_waveform = codec.preprocess(prompt_waveform, codec.sample_rate)

    # with torch.inference_mode():
    #     _, audio_tokens, _, _, _ = codec.encode(prompt_waveform)

    # audio_tokens = audio_tokens[:, :n_quantizers]
    # audio_tokens = audio_tokens + quantizer_offsets
    # audio_tokens = flatten_audio_tokens(audio_tokens)

    # audio_tokens = F.pad(audio_tokens, (1, 0), value=bos_token_id)

    # input_ids = torch.tensor([[bos_token_id]], dtype=torch.int64, device=device)

    frame_rate = 86.1
    max_duration = 8.0
    T = int(max_duration * frame_rate)

    amp_dtype = torch.bfloat16

    input_ids = tokenize(text)
    input_ids = rearrange(input_ids, "T -> 1 T").to(device)
    input_ids = F.pad(input_ids, (1, 0), value=bos_token_id)
    T_text = input_ids.size(-1)

    max_length = n_quantizers * T + T_text

    # with torch.amp.autocast(dtype=amp_dtype, device_type="cuda", enabled=True):
    out = decode(
        input_ids,
        model,
        max_length,
        n_quantizers=n_quantizers,
        text_offset=text_offset,
        top_k=top_k,
        temperature=temperature,
        cg=True,
    )

    print(f"{input_ids.shape=} {out.sequences.shape=}")

    audio_tokens = out.sequences[..., T_text:]
    audio_tokens = unflatten_audio_tokens(audio_tokens, T)

    print(f"{audio_tokens.shape=} {quantizer_offsets.shape=}")

    audio_tokens = audio_tokens - quantizer_offsets
    print(audio_tokens)

    # TODO should mask the logits out depending on which codebook we're decoding in decode
    audio_tokens = torch.where(audio_tokens < codebook_size, audio_tokens, 0)
    audio_tokens = torch.where(audio_tokens > 0, audio_tokens, 0)

    print(audio_tokens)

    with torch.inference_mode():
        z, _, _ = codec.quantizer.from_codes(audio_tokens)
        waveform = codec.decode(z)

    waveform = rearrange(waveform, "1 C T -> C T").cpu()
    path = f"{id}-{temperature=}-{top_k=}.wav"
    torchaudio.save(path, waveform, codec.sample_rate)

    print(f"Saved to {path}")

    return path


demo = gr.Interface(
    fn=generate,
    inputs=[
        # gr.Audio(type="filepath", label="prompt"),
        gr.Textbox(label="text", lines=5),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="temperature"),
        gr.Slider(minimum=1, maximum=1024, step=1, value=1024, label="top-k"),
    ],
    outputs=gr.Audio(),
)


demo.launch(debug=True)
