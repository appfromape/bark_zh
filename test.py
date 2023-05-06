import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchaudio
torchaudio.set_audio_backend("soundfile")

from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models(
    text_use_small=True,
    coarse_use_small=True,
    fine_use_gpu=True,
    fine_use_small=True,
)

# generate audio from text
text_prompt = """
各位观众朋友大家好. 这里是程式猿R法的频道.
首先很开心大家点击我这支影片.
我的影片主要是介绍一些实用的手机APP软体跟网站工具.
在这个平台分享给大家.
"""
audio_array = generate_audio(text_prompt, history_prompt="v2/zh_speaker_6")

# save audio to disk
write_wav("bark_6.wav", SAMPLE_RATE, audio_array)