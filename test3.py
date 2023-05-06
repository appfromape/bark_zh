import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torchaudio
torchaudio.set_audio_backend("soundfile")

from IPython.display import Audio
import nltk  # we'll use this to split into sentences
import numpy as np

nltk.download('punkt')

from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

import soundfile as sf

preload_models(
    text_use_small=True,
    coarse_use_small=True,
    fine_use_gpu=True,
    fine_use_small=True,
)

speaker_lookup = {"Samantha": "v2/zh_speaker_6", "John": "v2/zh_speaker_1"}

# Script generated by chat GPT
script = """
    Samantha: 嘿，你听说过这种叫做“bark”的新文本到音频模型吗？

    John: 不，我没有。 它有什么特别之处？

    Samantha: 好吧，显然它是目前最真实、听起来最自然的文本到音频模型。 人们说这听起来就像一个真人在说话。

    John: 哇，这听起来很棒。 它是如何工作的？

    Samantha: 我认为它使用先进的机器学习算法来分析和理解人类语音的细微差别，然后在自己的语音输出中复制这些细微差别。

    John: 这令人印象深刻。 你认为它可以用于有声读物或播客之类的东西吗？

    Samantha: 确实！ 事实上，我听说一些出版商已经开始使用 Bark 来制作有声读物。 我敢打赌这对播客也很棒。

    John: 我能想象。 这就像拥有自己的个人画外音艺术家一样。

    Samantha: 确切地！ 我认为 Bark 将成为文本到音频技术领域的游戏规则改变者。"""
script = script.strip().split("\n")
script = [s.strip() for s in script if s]
script

pieces = []
silence = np.zeros(int(0.5*SAMPLE_RATE))
for line in script:
    speaker, text = line.split(": ")
    audio_array = generate_audio(text, history_prompt=speaker_lookup[speaker], )
    pieces += [audio_array, silence.copy()]

# Combine all the audio pieces into a single array
final_audio = np.concatenate(pieces)

# Save the generated audio to a WAV file
sf.write('output_audio2.wav', final_audio, SAMPLE_RATE)