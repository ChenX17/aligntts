'''
Date: 2021-01-21 13:07:23
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-16 10:58:55
'''
import os
import librosa
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf
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

set_list = ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean']
# set_list = ['test-clean']
#csv_file = '/data/nfs_rt22/TTS/chenxi/data/LibriSpeech'
root_dir = '/data/nfs_rt22/TTS/chenxi/data/LibriSpeech'
data_dir = 'data/librispeech/preprocessed'
print("stage 1 read the label files")

csv_file_list = []
for dataset in set_list:
    set_dir = os.path.join(root_dir, dataset)
    pre_pre_csv_list = os.listdir(set_dir)
    for pre_pre_csv in pre_pre_csv_list:
        if not pre_pre_csv.isdigit():
            continue
        pre_csv_dir = os.path.join(set_dir,pre_pre_csv)
        pre_csv_list = os.listdir(pre_csv_dir)
        for pre_csv in pre_csv_list:
            csv_dir = os.path.join(pre_csv_dir, pre_csv)
            csv_list = os.listdir(csv_dir)
            for item in csv_list:
                if item.endswith('.txt'):
                    csv_file_list.append(os.path.join(csv_dir,item))

g2p = G2p()
metadata={}
#import pdb;pdb.set_trace()
print("stage 2 get the char seq and phone seq")
count = 0
all_label = {}
for csv_file in csv_file_list:
    with codecs.open(csv_file, 'r', 'utf-8') as fid:
        #import pdb;pdb.set_trace()
        for line in fid.readlines():
            count+=1
            if count % 1000 == 0:
                print('stage 2 processed %d sentences'%count)
            id, text = line.split(" ", 1)
            id = '/'.join(csv_file.split('/')[:-1])+'/'+id
            # import pdb;pdb.set_trace()
            
            clean_char = custom_english_cleaners(text.rstrip())
            clean_phone = []
            for s in g2p(clean_char.lower()):
                # import pdb;pdb.set_trace()
                if '@'+s in symbol_to_id:
                    clean_phone.append('@'+s)
                else:
                    clean_phone.append(s)
            # import pdb;pdb.set_trace()
            metadata[id]={'char':clean_char,
                        'phone':clean_phone}

from layers import TacotronSTFT
stft = TacotronSTFT(sampling_rate=16000)

def text2seq(text):
    sequence=[symbol_to_id['^']]
    sequence.extend([symbol_to_id[c] for c in text])
    sequence.append(symbol_to_id['~'])
    return sequence


def get_mel(filename):
    # import pdb;pdb.set_trace()
    # wav, sr = librosa.load(filename, sr=16000)
    wav, sr = sf.read(filename)
    wav = torch.FloatTensor(wav.astype(np.float32))
    melspec = stft.mel_spectrogram(wav.unsqueeze(0))
    return melspec.squeeze(0).numpy(), wav

fname_list = []
def save_file(fname):
    fname_list.append(fname.split('/')[-1])
    if len(fname_list) > len(list(set(fname_list))):
        import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
    wav_name = os.path.join(root_dir, fname) + '.flac'
    text = metadata[fname]['char']
    char_seq = np.asarray( text2seq(metadata[fname]['char']), dtype=np.int64 )
    try:
        phone_seq = np.asarray( text2seq(metadata[fname]['phone']), dtype=np.int64)
    except:
        phone_seq = np.asarray( text2seq([phone.replace('..', '.') for phone in metadata[fname]['phone']]), dtype=np.int64)
    
    melspec, wav = get_mel(wav_name)

    fname = fname.split('/')[-1]
    
    # Skip existing files
    # if os.path.isfile(f'{data_dir}/char_seq/{fname}_sequence.npy') and \
    #     os.path.isfile(f'{data_dir}/phone_seq/{fname}_sequence.npy') and \
    #     os.path.isfile(f'{data_dir}/melspectrogram/{fname}_melspectrogram.npy'):
    #     return text, char_seq, phone_seq, melspec, wav
    
    # np.save(f'{data_dir}/char_seq/{fname}_sequence.npy', char_seq)
    # np.save(f'{data_dir}/phone_seq/{fname}_sequence.npy', phone_seq)
    np.save(f'{data_dir}/melspectrogram/{fname}_melspectrogram.npy', melspec)

    return text, char_seq, phone_seq, melspec, wav
    
print("stage 3 extract mel feature")
mel_values = []

# set_list = ['train-clean-100', 'train-clean-360', 'dev-clean', 'test-clean']
train_files_list = []
val_files_list = []
test_files_list = []
for k in metadata.keys():
    if 'train' in k:
        train_files_list.append(k.split('/')[-1]+'|'+metadata[k]['char'])
    elif 'test' in k:
        test_files_list.append(k.split('/')[-1]+'|'+metadata[k]['char'])
    else:
        val_files_list.append(k.split('/')[-1]+'|'+metadata[k]['char'])

train_filelist_path = 'filelists/librispeech_audio_text_train_filelist.txt'
val_filelist_path = 'filelists/librispeech_audio_text_val_filelist.txt'
test_filelist_path = 'filelists/librispeech_audio_text_test_filelist.txt'

# f = open(train_filelist_path, 'w')
# f.writelines('\n'.join(train_files_list))
# f.close()

# f = open(val_filelist_path, 'w')
# f.writelines('\n'.join(val_files_list))
# f.close()

# f = open(test_filelist_path, 'w')
# f.writelines('\n'.join(test_files_list))
# f.close()


        
    
count = 0
for k in metadata.keys():
    count+=1
    if count % 1000 == 0:
        print('stage 3 processed %d sentences'%count)
    text, char_seq, phone_seq, melspec, wav = save_file(k)
