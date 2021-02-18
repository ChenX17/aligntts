import os, argparse
# os.environ["CUDA_VISIBLE_DEVICES"]='1'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('waveglow/')

# import IPython.display as ipd
import pickle as pkl
import torch
import torch.nn.functional as F
import torch.nn as nn
import hparams
from torch.utils.data import DataLoader
from modules.model_dnn import Model
from text import text_to_sequence, sequence_to_text_b
from denoiser import Denoiser
from tqdm import tqdm
import librosa
from modules.loss import MDNLoss
import math
import numpy as np
from datetime import datetime
from ctc import ctc
import matplotlib.pyplot as plt

def main():
    # data_type = 'phone'\
    data_type = 'char'
    checkpoint_path = "training_log/aligntts_am_char/stage0/checkpoint_40000"
    print(checkpoint_path)
    state_dict = {}
    
    for k, v in torch.load(checkpoint_path)['state_dict'].items():
        state_dict[k[7:]]=v

    model = Model(hparams).cuda()
    model.load_state_dict(state_dict)
    _ = model.cuda().eval()
    criterion = MDNLoss()
    
    datasets = ['val', 'train', 'test']
    batch_size=64

    for dataset in datasets:
        with open(f'filelists/ljs_audio_text_{dataset}_filelist.txt', 'r', encoding='utf-8') as f:
            lines_raw = [line.split('|') for line in f.read().splitlines()]
            lines_list = [ lines_raw[batch_size*i:batch_size*(i+1)] 
                          for i in range(len(lines_raw)//batch_size+1)]

        for batch in tqdm(lines_list):
            file_list, text_list, mel_list = [], [], []
            text_lengths, mel_lengths=[], []

            for i in range(len(batch)):
                file_name, _, text = batch[i]
                file_list.append(file_name)
                seq = os.path.join('data/LJSpeech-1.1/preprocessed',
                                   f'{data_type}_seq')
                mel = os.path.join('data/LJSpeech-1.1/preprocessed',
                                   'melspectrogram')

                seq = torch.from_numpy(np.load(f'{seq}/{file_name}_sequence.npy'))
                mel = torch.from_numpy(np.load(f'{mel}/{file_name}_melspectrogram.npy'))

                text_list.append(seq)
                mel_list.append(mel)
                text_lengths.append(seq.size(0))
                mel_lengths.append(mel.size(1))

            text_lengths = torch.LongTensor(text_lengths)
            mel_lengths = torch.LongTensor(mel_lengths)
            text_padded = torch.zeros(len(batch), text_lengths.max().item(), dtype=torch.long)
            mel_padded = torch.zeros(len(batch), hparams.n_mel_channels, mel_lengths.max().item())

            for j in range(len(batch)):
                text_padded[j, :text_list[j].size(0)] = text_list[j]
                mel_padded[j, :, :mel_list[j].size(1)] = mel_list[j]

            text_padded = text_padded.cuda()
            mel_padded = mel_padded.cuda()
            #import pdb;pdb.set_trace()
            mel_padded = (torch.clamp(mel_padded, hparams.min_db, hparams.max_db)-hparams.min_db) / (hparams.max_db-hparams.min_db)
            text_lengths = text_lengths.cuda()
            mel_lengths = mel_lengths.cuda()

            with torch.no_grad():
                # encoder_input = model.Prenet(text_padded)
                # hidden_states, _ = model.FFT_lower(encoder_input, text_lengths)
                log_probs, hidden_states_spec, logits = model.get_am(mel_padded, mel_lengths, text_padded)
                import pdb;pdb.set_trace()
                decode_results = ctc.ctc_decode(logits, mel_lengths, blank=119)
                # mu_sigma = model.get_mu_sigma(hidden_states)
                # _, log_prob_matrix = criterion(mu_sigma, mel_padded, text_lengths, mel_lengths)
                # import pdb;pdb.set_trace()
                # align = ctc.ctc_alignment(log_probs, text_padded, mel_lengths, text_lengths, blank = 119)
                align = ctc.ctc_alignment(torch.transpose(logits, 0, 1), text_padded, mel_lengths, text_lengths, blank = 119)
            results = np.array(decode_results)
            for k in range(len(batch)):
                alignment_with_len = torch.cat([align[k][:text_lengths[k]], mel_lengths[k].view(-1)], 0)
                alignment = alignment_with_len[1:] - alignment_with_len[:-1]
                # import pdb;pdb.set_trace()
                np.save(f'data/LJSpeech-1.1/preprocessed/alignments_am_chars/{file_list[k]}_alignment.npy',
                    alignment.detach().cpu().numpy())
                np.save(f'data/LJSpeech-1.1/preprocessed/decode_am_chars/{file_list[k]}_seq.npy', results[k][1:-1])


                tmp_align = alignment.detach().cpu().numpy()
                y = []
                for i in range(len(tmp_align)):
                    y.append(sum(tmp_align[:i+1]))
            

                x = [i for i in range(len(tmp_align))]
                plt.plot(x, y)
                plt.savefig("alignment_ctc.png", dpi=120)
                

                # align = model.viterbi(log_prob_matrix, text_lengths, mel_lengths).to(torch.long)
            #     alignments = list(torch.split(align,1))

            # for j, (l, t) in enumerate(zip(text_lengths, mel_lengths)):
            #     # import pdb;pdb.set_trace()
            #     alignments[j] = alignments[j][0, :l.item(), :t.item()].sum(dim=-1)
            #     np.save(f'data/LJSpeech-1.1/preprocessed/alignments/{file_list[j]}_alignment.npy',
            #             alignments[j].detach().cpu().numpy())
            
    print("Alignments Extraction End!!! ({datetime.now()})")
          
        
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('-v', '--verbose', type=str, default='0')
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.verbose=='0':
        import warnings
        warnings.filterwarnings("ignore")
        
    main()