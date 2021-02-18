'''
Date: 2021-02-18 10:39:11
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-18 11:10:18
'''
import matplotlib.pyplot as plt
import numpy as np

mel_path = '/data/nfs_rt22/TTS/chenxi/projects/AlignTTS/data/librispeech/preprocessed/melspectrogram/4731-58193-0009_melspectrogram.npy'

mel = np.load(mel_path)
import pdb;pdb.set_trace()

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

im = ax1.imshow(mel, aspect="auto", interpolation="none")
ax1.set_title("Predicted Mel-before-Spectrogram")
fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)

im = ax2.imshow(mel, aspect="auto", interpolation="none")
ax2.set_title("Predicted Mel-after-Spectrogram")
fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
plt.tight_layout()
plt.savefig("mel.png")
plt.close()

# plt.plot(mel)
# plt.savefig("mel.png")
