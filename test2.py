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

script = """
    嘿 你听说过这种叫做"bark"的新文字到音频模型吗?
    显然 这是最真实 听起来最自然的文本到音频模型.
    现在就在那里 人们说这听起来就像一个真人在说话.
    我认为它使用先进的机器学习算法来分析和理解.
    人类语音的细微差别 然后在自己的语音输出中复制这些细微差别.
    它非常令人印象深刻 我敢打赌它可以用于有声读物或播客之类的东西.
    事实上 我听说一些出版商已经开始使用 Bark 来制作有声读物.
    这就像拥有自己的个人画外音艺术家一样.我真的认为bark会
    成为文字到音频技术领域的游戏规则改变者.
""".replace("\n", " ").strip()

sentences = nltk.sent_tokenize(script)

SPEAKER = "v2/zh_speaker_6"
silence = np.zeros(int(0.25 * SAMPLE_RATE))  # quarter second of silence

pieces = []
for sentence in sentences:
    audio_array = generate_audio(sentence, history_prompt=SPEAKER)
    pieces += [audio_array, silence.copy()]

# Combine all the audio pieces into a single array
final_audio = np.concatenate(pieces)

# Save the generated audio to a WAV file
sf.write('output_audio.wav', final_audio, SAMPLE_RATE)