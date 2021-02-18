'''
Date: 2021-01-21 13:07:23
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-17 21:33:22
'''
import os
import librosa
from librosa.filters import mel as librosa_mel_fn
import pickle as pkl
# import IPython.display as ipd
from tqdm.notebook import tqdm
import torch
import numpy as np
import codecs
import matplotlib.pyplot as plt

from g2p_en import G2p
from text import *
from text import cmudict
from text.cleaners import custom_english_cleaners
from text.symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
symbol_to_id = {s: i for i, s in enumerate(symbols)}
id_to_symbol = {i: s for i, s in enumerate(symbols)}


csv_file = 'data/LJSpeech-1.1/metadata.csv'
root_dir = 'data/LJSpeech-1.1/wavs'
data_dir = 'data/LJSpeech-1.1/preprocessed'

g2p = G2p()
metadata={}
with codecs.open(csv_file, 'r', 'utf-8') as fid:
    for line in fid.readlines():
        id, _, text = line.split("|")
        
        clean_char = custom_english_cleaners(text.rstrip())
        clean_phone = []
        for s in g2p(clean_char.lower()):
            if '@'+s in symbol_to_id:
                clean_phone.append('@'+s)
            else:
                clean_phone.append(s)

        metadata[id]={'char':clean_char,
                     'phone':clean_phone}

from layers import TacotronSTFT
stft = TacotronSTFT()

def text2seq(text):
    # sequence=[symbol_to_id['^']]
    sequence = []
    sequence.extend([symbol_to_id[c]+1 for c in text])
    # sequence.append(symbol_to_id['~'])
    return sequence


def get_mel(filename):
    wav, sr = librosa.load(filename, sr=22050)
    wav = torch.FloatTensor(wav.astype(np.float32))
    melspec = stft.mel_spectrogram(wav.unsqueeze(0))
    return melspec.squeeze(0).numpy(), wav

def save_file(fname):
        
    wav_name = os.path.join(root_dir, fname) + '.wav'
    text = metadata[fname]['char']
    char_seq = np.asarray( text2seq(metadata[fname]['char']), dtype=np.int64 )
    try:
        phone_seq = np.asarray( text2seq(metadata[fname]['phone']), dtype=np.int64)
    except:
        phone_seq = np.asarray( text2seq([phone.replace('..', '.') for phone in metadata[fname]['phone']]), dtype=np.int64)
    
    melspec, wav = get_mel(wav_name)
    
    # Skip existing files
    # if os.path.isfile(f'{data_dir}/char_seq/{fname}_sequence.npy') and \
    #     os.path.isfile(f'{data_dir}/phone_seq/{fname}_sequence.npy') and \
    #     os.path.isfile(f'{data_dir}/melspectrogram/{fname}_melspectrogram.npy'):
    #     return text, char_seq, phone_seq, melspec, wav
    
    np.save(f'{data_dir}/char_seq/{fname}_sequence.npy', char_seq)
    np.save(f'{data_dir}/phone_seq/{fname}_sequence.npy', phone_seq)
    np.save(f'{data_dir}/melspectrogram/{fname}_melspectrogram.npy', melspec)

    return text, char_seq, phone_seq, melspec, wav
    
mel_values = []
for k in metadata.keys():
    text, char_seq, phone_seq, melspec, wav = save_file(k)
