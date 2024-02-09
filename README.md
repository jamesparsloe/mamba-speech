# MambaSpeech

Trying to get Mamba to speak with the help of [DAC](https://github.com/descriptinc/descript-audio-codec).

```sh
python3.11 -m venv env
source env/bin/activate

python -m pip install wheel packaging
python -m pip install -e .
python -m pip install "causal-conv1d>=1.1.0"
python -m pip install mamba-ssm


pip install flash-attn --no-build-isolation
```

