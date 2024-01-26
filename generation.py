from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel
from mamba_ssm.utils.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
    InferenceParams,
    modify_logits_for_top_p_filtering,
    modify_logit_for_repetition_penalty,
)
import json
import torch
from torch import Tensor
from einops import rearrange
import dac
import torchaudio
import time


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

    batch_size, seqlen_og = input_ids.shape
    teacher_output_len = 0

    inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

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

    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], inference_params))
        inference_params.seqlen_offset += sequences[-1].shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1], inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)

    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")

    output_cls = (
        GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    )

    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


if __name__ == "__main__":
    device = "cuda"

    dac_path = dac.utils.download(model_type="44khz")
    codec = dac.DAC.load(dac_path).eval().to(device)

    config_path = "runs/07gd8g19/002000/config.json"
    checkpoint_path = "runs/07gd8g19/002000/pytorch_model.bin"

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

    n = 10
    id = int(time.time())
    top_k = 64
    temperature = 1.0

    for i in range(n):
        input_ids = torch.tensor([[bos_token_id]], dtype=torch.int64, device=device)

        frame_rate = 86.1
        T = int(frame_rate)
        max_length = n_quantizers * T + 1  # we'll knock off the bos token

        amp_dtype = torch.bfloat16

        # with torch.amp.autocast(dtype=amp_dtype, device_type="cuda", enabled=True):
        out = decode(input_ids, model, max_length, top_k=top_k, temperature=temperature)

        audio_tokens = out.sequences[..., 1:]
        audio_tokens = unflatten_audio_tokens(audio_tokens, T)

        quantizer_offsets = codebook_size * torch.arange(n_quantizers, device=device)
        quantizer_offsets = rearrange(quantizer_offsets, "Q -> Q 1")

        audio_tokens = audio_tokens - quantizer_offsets
        print(audio_tokens)

        # TODO should mask the logits out depending on which codebook we're decoding in decode
        audio_tokens = torch.where(audio_tokens < 1024, audio_tokens, 0)
        audio_tokens = torch.where(audio_tokens > 0, audio_tokens, 0)

        print(audio_tokens)

        with torch.inference_mode():
            z, _, _ = codec.quantizer.from_codes(audio_tokens)
            waveform = codec.decode(z)

        waveform = rearrange(waveform, "1 C T -> C T").cpu()
        torchaudio.save(
            f"{id}-{i}-{temperature=}-{top_k=}.wav", waveform, codec.sample_rate
        )

    # print(generated.shape)
