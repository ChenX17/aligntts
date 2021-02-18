'''
Date: 2021-01-22 10:14:47
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-02 23:32:57
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('waveglow/')

import pickle as pkl
import torch
import torch.nn.functional as F
import hparams
from torch.utils.data import DataLoader
from modules.model import Model
from text import text_to_sequence, sequence_to_text
from denoiser import Denoiser
from tqdm import tqdm_notebook as tqdm
import librosa
from modules.loss import MDNLoss
import math
from multiprocessing import Pool
import numpy as np
from text.symbols import symbols
import matplotlib.pyplot as plt

data_type_ = 'char'
data_type = 'phone'
checkpoint_path = f"training_log/aligntts/stage0/checkpoint_40000"

from glob import glob

# checkpoint_path = sorted(glob("training_log/aligntts/stage0/checkpoint_*"))[0]
checkpoint_path = "training_log/aligntts/stage0/checkpoint_40000"

print(checkpoint_path)

state_dict = {}
for k, v in torch.load(checkpoint_path)['state_dict'].items():
    state_dict[k[7:]]=v


model = Model(hparams).cuda()
model.load_state_dict(state_dict)
_ = model.cuda().eval()
criterion = MDNLoss()

import time

datasets = ['train', 'val', 'test']
batch_size=64
batch_size = 1

start = time.perf_counter()

for dataset in datasets:
    
    with open(f'filelists/ljs_audio_text_{dataset}_filelist.txt', 'r', encoding='utf-8') as f:
        lines_raw = [line.split('|') for line in f.read().splitlines()]
        lines_list = [ lines_raw[batch_size*i:batch_size*(i+1)] 
                      for i in range(len(lines_raw)//batch_size+1)]
        
    for batch in lines_list:
        single_loop_start = time.perf_counter()
        
        file_list, text_list, mel_list = [], [], []
        text_list_ = []
        text_lengths, mel_lengths=[], []
        text_lengths_ = []
        
        for i in range(len(batch)):
            file_name, _, text = batch[i]
            file_list.append(file_name)
            seq_path = os.path.join('data/LJSpeech-1.1/preprocessed',
                               f'{data_type}_seq')
            seq_path_ = os.path.join('data/LJSpeech-1.1/preprocessed',
                               f'{data_type_}_seq')
            mel_path = os.path.join('data/LJSpeech-1.1/preprocessed',
                               'melspectrogram')
            try:
                seq = torch.from_numpy(np.load(f'{seq_path}/{file_name}_sequence.npy'))
                seq_ = torch.from_numpy(np.load(f'{seq_path_}/{file_name}_sequence.npy'))
            except FileNotFoundError:
                with open(f'{seq_path}/{file_name}_sequence.pkl', 'rb') as f:
                    seq = pkl.load(f)
            try:
                mel = torch.from_numpy(np.load(f'{mel_path}/{file_name}_melspectrogram.npy'))
            except FileNotFoundError:
                with open(f'{mel_path}/{file_name}_melspectrogram.pkl', 'rb') as f:
                    mel = pkl.load(f)
            # import pdb;pdb.set_trace()
            text_list.append(seq)
            text_list_.append(seq_)
            mel_list.append(mel)
            text_lengths.append(seq.size(0))
            text_lengths_.append(seq_.size(0))
            mel_lengths.append(mel.size(1))

        io_time = time.perf_counter()
            
        text_lengths = torch.LongTensor(text_lengths)
        text_lengths_ = torch.LongTensor(text_lengths_)
        mel_lengths = torch.LongTensor(mel_lengths)
        text_padded = torch.zeros(len(batch), text_lengths.max().item(), dtype=torch.long)
        text_padded_ = torch.zeros(len(batch), text_lengths_.max().item(), dtype=torch.long)
        mel_padded = torch.zeros(len(batch), hparams.n_mel_channels, mel_lengths.max().item())

        for j in range(len(batch)):
            text_padded[j, :text_list[j].size(0)] = text_list[j]
            text_padded_[j, :text_list_[j].size(0)] = text_list_[j]
            mel_padded[j, :, :mel_list[j].size(1)] = mel_list[j]
        
        text_padded = text_padded.cuda()
        text_padded_ = text_padded_.cuda()
        mel_padded = mel_padded.cuda()
        mel_padded = (torch.clamp(mel_padded, hparams.min_db, hparams.max_db)-hparams.min_db) / (hparams.max_db-hparams.min_db)
        text_lengths = text_lengths.cuda()
        text_lengths_ = text_lengths_.cuda()
        mel_lengths = mel_lengths.cuda()

        with torch.no_grad():
            
            model_start = time.perf_counter()
            
            encoder_input = model.Prenet(text_padded)
            hidden_states, _ = model.FFT_lower(encoder_input, text_lengths)
            mu_sigma = model.get_mu_sigma(hidden_states)
            _, log_prob_matrix = criterion(mu_sigma, mel_padded, text_lengths, mel_lengths)
            
            viterbi_start = time.perf_counter()

            align = model.viterbi(log_prob_matrix, text_lengths, mel_lengths).to(torch.long)
            alignments = list(torch.split(align,1))
            
            viterbi_end = time.perf_counter()
        
        print('VT Time: ', end=' ')
        print(f'{viterbi_end - viterbi_start:.6f} / {viterbi_end - single_loop_start:.6f} = ' +
             f'{(viterbi_end - viterbi_start) / (viterbi_end - single_loop_start) * 100:5.2f}%')

        print('IO Time: ', end=' ')
        print(f'{io_time - single_loop_start:.6f} / {viterbi_end - single_loop_start:.6f} = ' +
             f'{(io_time - single_loop_start) / (viterbi_end - single_loop_start) * 100:5.2f}%')
        
        print('DL Time: ', end=' ')
        print(f'{viterbi_start - model_start:.6f} / {viterbi_end - single_loop_start:.6f} = ' +
             f'{(viterbi_start - model_start) / (viterbi_end - single_loop_start) * 100:5.2f}%')


        print(alignments[0].shape)
        # import pdb;pdb.set_trace()
        
        # break
        for j, (l, t) in enumerate(zip(text_lengths, mel_lengths)):
            alignments[j] = alignments[j][0, :l.item(), :t.item()].sum(dim=-1)
            np.save(f'data/LJSpeech-1.1/preprocessed/alignments/{file_list[j]}_alignment.npy',
                    alignments[j].detach().cpu().numpy())
            tmp_align = alignments[j].detach().cpu().numpy()
            y = []
            for i in range(len(tmp_align)):
                y.append(sum(tmp_align[:i+1]))
            

            x = [i for i in range(len(tmp_align))]
            plt.plot(x, y)
            plt.savefig("alignment.png", dpi=120)
    
        with torch.no_grad():
            
            model_start = time.perf_counter()
            
            encoder_input = model.Prenet(text_padded_)
            hidden_states, _ = model.FFT_lower(encoder_input, text_lengths_)
            mu_sigma = model.get_mu_sigma(hidden_states)
            # import pdb;pdb.set_trace()
            _, log_prob_matrix = criterion(mu_sigma, mel_padded, text_lengths_, mel_lengths)
            
            viterbi_start = time.perf_counter()

            align = model.viterbi(log_prob_matrix, text_lengths_, mel_lengths).to(torch.long)
            alignments_ = list(torch.split(align,1))
            
            viterbi_end = time.perf_counter()
        
        print('VT Time: ', end=' ')
        print(f'{viterbi_end - viterbi_start:.6f} / {viterbi_end - single_loop_start:.6f} = ' +
             f'{(viterbi_end - viterbi_start) / (viterbi_end - single_loop_start) * 100:5.2f}%')

        print('IO Time: ', end=' ')
        print(f'{io_time - single_loop_start:.6f} / {viterbi_end - single_loop_start:.6f} = ' +
             f'{(io_time - single_loop_start) / (viterbi_end - single_loop_start) * 100:5.2f}%')
        
        print('DL Time: ', end=' ')
        print(f'{viterbi_start - model_start:.6f} / {viterbi_end - single_loop_start:.6f} = ' +
             f'{(viterbi_start - model_start) / (viterbi_end - single_loop_start) * 100:5.2f}%')


        print(alignments_[0].shape)
        # import pdb;pdb.set_trace()
        
        # break
        for j, (l, t) in enumerate(zip(text_lengths, mel_lengths)):
            alignments_[j] = alignments_[j][0, :l.item(), :t.item()].sum(dim=-1)
            np.save(f'data/LJSpeech-1.1/preprocessed/alignments/{file_list[j]}_alignment.npy',
                    alignments[j].detach().cpu().numpy())
            
            
            # plt.show()



        