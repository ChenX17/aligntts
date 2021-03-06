'''
Date: 2021-01-23 18:37:19
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-02 23:30:55
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import hparams as hp
from utils.utils import get_mask_from_lengths
import math



class MDNLoss(nn.Module):
    def __init__(self):
        super(MDNLoss, self).__init__()
        
    def forward(self, mu_sigma, melspec, text_lengths, mel_lengths):
        # mu, sigma: B, L, F / melspec: B, F, T
        B, L, _ = mu_sigma.size()
        T = melspec.size(2)
        
        x = melspec.transpose(1,2).unsqueeze(1) # B, 1, T, F
        mu = torch.sigmoid(mu_sigma[:, :, :hp.n_mel_channels].unsqueeze(2)) # B, L, 1, F
        log_sigma = mu_sigma[:, :, hp.n_mel_channels:].unsqueeze(2) # B, L, 1, F
    
        exponential = -0.5*torch.sum((x-mu)*(x-mu)/log_sigma.exp()**2, dim=-1) # B, L, T
        log_prob_matrix = exponential - (hp.n_mel_channels/2)*torch.log(torch.tensor(2*math.pi)) - 0.5 * log_sigma.sum(dim=-1)
        log_alpha = mu_sigma.new_ones(B, L, T)*(-1e30)
        log_alpha[:, 0, 0] = log_prob_matrix[:,0, 0]

        # import pdb;pdb.set_trace()
        # prob_matrix = torch.tensor(2*math.pi).exp()**(-0.5) * torch.exp(-0.5*torch.sum((x-mu)*(x-mu)/log_sigma.exp()**2, dim=-1))
        # alpha = mu_sigma.new_ones(B, L, T)*(1e-30)
        # alpha[:, 0, 0] = prob_matrix[:,0, 0]

        # import pdb;pdb.set_trace()
        
        for t in range(1, T):
            prev_step = torch.cat([log_alpha[:, :, t-1:t], F.pad(log_alpha[:, :, t-1:t], (0,0,1,-1), value=-1e30)], dim=-1)
            log_alpha[:, :, t] = torch.logsumexp(prev_step+1e-30, dim=-1)+log_prob_matrix[:, :, t]
            # prev_step = torch.cat([alpha[:, :, t-1:t], F.pad(alpha[:, :, t-1:t], (0,0,1,-1), value=1e-30)], dim=-1)
            # alpha[:, :, t] = torch.sum(prev_step+1e-30, dim=-1)*prob_matrix[:, :, t]

            
            # scaler = torch.unsqueeze(1 / torch.sum(log_alpha[:, :, t], dim=1) + 1e-30, -1)
            # log_alpha[:, :, t] = log_alpha[:, :, t] * scaler
        
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1]
        # alpha_last = torch.log(alpha[torch.arange(B), text_lengths-1, mel_lengths-1])

        mdn_loss = -alpha_last.mean()

        return mdn_loss, log_prob_matrix

class MDNDNNLoss(nn.Module):
    def __init__(self):
        super(MDNDNNLoss, self).__init__()
        
    def forward(self, probs, melspec, text_lengths, mel_lengths):
        # mu, sigma: B, L, F / melspec: B, F, T
        # B, L, _ = mu_sigma.size()
        # probs: B, L, T
        # import pdb;pdb.set_trace()
        B, L, _ = probs.size()
        T = melspec.size(2)
        
        # x = melspec.transpose(1,2).unsqueeze(1) # B, 1, T, F
        # mu = torch.sigmoid(mu_sigma[:, :, :hp.n_mel_channels].unsqueeze(2)) # B, L, 1, F
        # log_sigma = mu_sigma[:, :, hp.n_mel_channels:].unsqueeze(2) # B, L, 1, F
    
        # exponential = -0.5*torch.sum((x-mu)*(x-mu)/log_sigma.exp()**2, dim=-1) # B, L, T
        log_prob_matrix = torch.log(probs+1e-30)
        log_alpha = probs.new_ones(B, L, T)*(-1e30)
        log_alpha[:, 0, 0] = log_prob_matrix[:,0, 0]

        
        for t in range(1, T):
            prev_step = torch.cat([log_alpha[:, :, t-1:t], F.pad(log_alpha[:, :, t-1:t], (0,0,1,-1), value=-1e30)], dim=-1)
            log_alpha[:, :, t] = torch.logsumexp(prev_step+1e-30, dim=-1)+log_prob_matrix[:, :, t]
            # prev_step = torch.cat([alpha[:, :, t-1:t], F.pad(alpha[:, :, t-1:t], (0,0,1,-1), value=1e-30)], dim=-1)
            # alpha[:, :, t] = torch.sum(prev_step+1e-30, dim=-1)*prob_matrix[:, :, t]

            
            # scaler = torch.unsqueeze(1 / torch.sum(log_alpha[:, :, t], dim=1) + 1e-30, -1)
            # log_alpha[:, :, t] = log_alpha[:, :, t] * scaler
        
        alpha_last = log_alpha[torch.arange(B), text_lengths-1, mel_lengths-1]
        # alpha_last = torch.log(alpha[torch.arange(B), text_lengths-1, mel_lengths-1])

        mdn_loss = -alpha_last.mean()

        return mdn_loss, log_prob_matrix