import dac
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class AudioCodec(nn.Module):
    def __init__(self):
        super().__init__()
        path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(path)

        self.eval()

    @property
    def sample_rate(self):
        return self.model.sample_rate

    @torch.no_grad()
    def encode(self, waveforms: list[Tensor]):
        device = waveforms[0].device
        lengths = torch.tensor([w.size(-1) for w in waveforms], device=device)
        waveforms = pad_sequence(waveforms, batch_first=True)

        x = self.model.preprocess(waveforms, self.sample_rate)
        z, codes, latents, _, _ = self.encode(x)

    @torch.no_grad()
    def decode(self, codes: Tensor): ...


if __name__ == "__main__":
    import torchaudio
    from torchdata.datapipes.iter import HuggingFaceHubReader
    from torchdata.datapipes.map import SequenceWrapper

    from .constants import CACHE_DIR
    from .utils import to_iter_datapipe

    # ds = torchaudio.datasets.LJSPEECH(CACHE_DIR, download=True)

    ds = torchaudio.datasets.SPEECHCOMMANDS(CACHE_DIR, download=True)
    dp = to_iter_datapipe(ds)
    dp_iter = iter(dp)
    for i in range(100):
        waveform, sample_rate, label, speaker_id, utterance_num = next(dp_iter)
        torchaudio.save(
            f"{label=}-{speaker_id=}-{utterance_num=}.wav", waveform, sample_rate
        )
        print(waveform.shape)
    # codec = AudioCodec()
