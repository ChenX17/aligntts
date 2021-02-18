import os, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from ctc import ctc
# from modules.model import Model
from modules.model_dnn import Model
from modules.loss import MDNLoss,MDNDNNLoss
# import hparams_librispeech as hparams
import hparams
from text import *
from utils.utils import *
from utils.writer import get_writer
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
import numpy as np

import pdb;pdb.set_trace()


def validate(model, criterion, val_loader, iteration, writer, stage):
    model.eval()
    with torch.no_grad():
        n_data, val_loss = 0, 0
        for i, batch in enumerate(val_loader):
            n_data += len(batch[0])
            if stage==0:
                text_padded, mel_padded, text_lengths, mel_lengths = [
                    reorder_batch(x, hparams.n_gpus).cuda() for x in batch
                ]
            else:
                text_padded, mel_padded, align_padded, text_lengths, mel_lengths = [
                    reorder_batch(x, hparams.n_gpus).cuda() for x in batch
                ]
                
            if stage !=3:
                encoder_input = model.module.Prenet(text_padded)
                hidden_states, _ = model.module.FFT_lower(encoder_input, text_lengths)
                
                if stage==0:
                    log_probs, hidden_states_spec, logits = model.module.get_am(mel_padded, mel_lengths, text_padded)
                    # import pdb;pdb.set_trace()
                    # np.save('logits_array_1.npy', logits.cpu().numpy())
                    # np.save('probs_array_1.npy', torch.exp(log_probs).cpu().numpy())
                    # align = ctc.ctc_alignment(torch.transpose(logits, 0, 1), text_padded, mel_lengths, text_lengths, blank = 119)
                    mel_lengths = torch.ceil(mel_lengths.float()/2).long()
                    loss = model.module.ctc_loss(log_probs, text_padded, mel_lengths, text_lengths)/log_probs.size(1)
                    # mu_sigma = model.module.get_mu_sigma(hidden_states)
                    # loss, log_prob_matrix = criterion(mu_sigma, mel_padded, text_lengths, mel_lengths)
                    
                elif stage==1:
                    mel_out = model.module.get_melspec(hidden_states, align_padded, mel_lengths)
                    mel_mask = ~get_mask_from_lengths(mel_lengths)
                    mel_padded_selected = mel_padded.masked_select(mel_mask.unsqueeze(1))
                    mel_out_selected = mel_out.masked_select(mel_mask.unsqueeze(1))
                    loss = nn.L1Loss()(mel_out_selected, mel_padded_selected)
                    
                elif stage==2:
                    # mu_sigma = model.module.get_mu_sigma(hidden_states)
                    probs = model.module.get_am(mel_padded, mel_lengths, text_padded)
                    mdn_loss, log_prob_matrix = criterion(probs, mel_padded, text_lengths, mel_lengths)
                    
                    align = model.module.viterbi(log_prob_matrix, text_lengths, mel_lengths)
                    mel_out = model.module.get_melspec(hidden_states, align, mel_lengths)
                    mel_mask = ~get_mask_from_lengths(mel_lengths)
                    mel_padded_selected = mel_padded.masked_select(mel_mask.unsqueeze(1))
                    mel_out_selected = mel_out.masked_select(mel_mask.unsqueeze(1))
                    fft_loss = nn.L1Loss()(mel_out_selected, mel_padded_selected)
                    loss = mdn_loss + fft_loss
                
            elif stage==3:
                duration_out = model.module.get_duration(text_padded, text_lengths) # gradient cut
                duration_target = align_padded.sum(-1)
                duration_mask = ~get_mask_from_lengths(text_lengths)
                duration_out = duration_out.masked_select(duration_mask)
                duration_target = duration_target.masked_select(duration_mask)
                loss = nn.MSELoss()(torch.log(duration_out), torch.log(duration_target))
              
            val_loss += loss.mean().item() * len(batch[0])

        val_loss /= n_data
        
    if stage==0:
        writer.add_scalar('Validation loss', val_loss, iteration//hparams.accumulation)
        # import pdb;pdb.set_trace()
        # align = ctc.ctc_alignment(log_probs, text_padded, mel_lengths, text_lengths, blank = 119)
        # import pdb;pdb.set_trace()
        # align = ctc.viterbi_align(logits[0].cpu().detach().numpy(), text_padded[0].view(-1).cpu().detach().numpy(), blank_id=119)
        align = ctc.viterbi_align(logits[0][:mel_lengths[0]].cpu().detach().numpy(), text_padded[0][:text_lengths[0]].view(-1).cpu().detach().numpy())
        import pdb;pdb.set_trace()
        align_plot = ctc.align_mask(align[0], mel_lengths[0])
        # hyps = ctc.ctc_decode(logits, mel_lengths, blank=119)
        # align = model.module.viterbi(log_prob_matrix[0:1], text_lengths[0:1], mel_lengths[0:1]) # 1, L, T
        # mel_out = torch.matmul(align[0].float().t(), mu_sigma[0, :, :hparams.n_mel_channels]).t() # F, T

        # writer.add_image('Validation_alignments', align.detach().cpu(), iteration//hparams.accumulation)
        # import pdb;pdb.set_trace()
        writer.add_image('Validation_alignments', align_plot, iteration//hparams.accumulation)
        # writer.add_specs(mel_padded[0].detach().cpu(),
        #                  mel_out.detach().cpu(),
        #                  iteration//hparams.accumulation, 'Validation')
    elif stage==1:
        writer.add_scalar('Validation loss', val_loss, iteration//hparams.accumulation)
        writer.add_specs(mel_padded[0].detach().cpu(),
                         mel_out[0].detach().cpu(),
                         iteration//hparams.accumulation, 'Validation')
    elif stage==2:
        writer.add_scalar('Validation mdn_loss', mdn_loss.item(), iteration//hparams.accumulation)
        writer.add_scalar('Validation fft_loss', fft_loss.item(), iteration//hparams.accumulation)
        writer.add_image('Validation_alignments',
                         align[0:1, :text_lengths[0], :mel_lengths[0]].detach().cpu(),
                         iteration//hparams.accumulation)
        writer.add_specs(mel_padded[0].detach().cpu(),
                         mel_out[0].detach().cpu(),
                         iteration//hparams.accumulation, 'Validation')
    elif stage==3:
        writer.add_scalar('Validation loss', val_loss, iteration//hparams.accumulation)
    
    model.train()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def main(args):
    train_loader, val_loader, collate_fn = prepare_dataloaders(hparams, stage=args.stage)
    initial_iteration = None
    if args.stage!=0 and args.pre_trained_model != '':
        checkpoint_path = f"training_log/aligntts/stage{args.stage-1}/checkpoint_{hparams.train_steps[args.stage-1]}"
        
        if not os.path.isfile(checkpoint_path):
            print(f'{checkpoint_path} does not exist')
            checkpoint_path = sorted(glob(f"training_log/aligntts/stage{args.stage-1}/checkpoint_*"))[-1]
            print(f'Loading {checkpoint_path} instead')
        
        state_dict = {}
        for k, v in torch.load(checkpoint_path)['state_dict'].items():
            state_dict[k[7:]]=v

        model = Model(hparams).cuda()
        model.load_state_dict(state_dict)
        model = nn.DataParallel(model).cuda()
    elif args.stage!=0:
        model = nn.DataParallel(Model(hparams)).cuda()
    else:
        if args.pre_trained_model != '':
            if not os.path.isfile(args.pre_trained_model):
                print(f'{args.pre_trained_model} does not exist')

            state_dict = {}
            for k, v in torch.load(args.pre_trained_model)['state_dict'].items():
                state_dict[k[7:]]=v
            initial_iteration = torch.load(args.pre_trained_model)['iteration']
            model = Model(hparams).cuda()
            model.load_state_dict(state_dict)
            model = nn.DataParallel(model).cuda()
        else:

            model = nn.DataParallel(Model(hparams)).cuda()

    criterion = MDNDNNLoss()
    writer = get_writer(hparams.output_directory, f'{hparams.log_directory}/stage{args.stage}')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams.lr,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    iteration, loss = 0, 0
    if initial_iteration is not None:
        iteration = initial_iteration

    model.train()

    print(f'Stage{args.stage} Start!!! ({str(datetime.now())})')
    while True:
        for i, batch in enumerate(train_loader):
            if args.stage==0:
                text_padded, mel_padded, text_lengths, mel_lengths = [
                    reorder_batch(x, hparams.n_gpus).cuda() for x in batch
                ]
                align_padded=None
            else:
                text_padded, mel_padded, align_padded, text_lengths, mel_lengths = [
                    reorder_batch(x, hparams.n_gpus).cuda() for x in batch
                ]

            sub_loss = model(text_padded,
                             mel_padded,
                             align_padded,
                             text_lengths,
                             mel_lengths,
                             criterion,
                             stage=args.stage,
                             log_viterbi=args.log_viterbi,
                             cpu_viterbi=args.cpu_viterbi)
            sub_loss = sub_loss.mean()/hparams.accumulation
            sub_loss.backward()
            loss = loss+sub_loss.item()
            iteration += 1
            if iteration % 100==0:
                print(f'[{str(datetime.now())}] Stage {args.stage} Iter {iteration:<6d} Loss {loss:<8.6f}')

            if iteration%hparams.accumulation == 0:
                # lr_scheduling(optimizer, iteration//hparams.accumulation)
                nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
                optimizer.step()
                model.zero_grad()
                writer.add_scalar('Train loss', loss, iteration//hparams.accumulation)
                writer.add_scalar('Learning rate', get_lr(optimizer), iteration//hparams.accumulation)
                loss=0
            

            # validate(model, criterion, val_loader, iteration, writer, args.stage)
            if iteration%(hparams.iters_per_validation*hparams.accumulation)==0:
                validate(model, criterion, val_loader, iteration, writer, args.stage)

            if iteration%(hparams.iters_per_checkpoint*hparams.accumulation)==0:
                save_checkpoint(model,
                                optimizer,
                                hparams.lr,
                                iteration//hparams.accumulation,
                                filepath=f'{hparams.output_directory}/{hparams.log_directory}/stage{args.stage}')

            if iteration==(hparams.train_steps[args.stage]*hparams.accumulation):
                break

        if iteration==(hparams.train_steps[args.stage]*hparams.accumulation):
            break
            
    print(f'Stage{args.stage} End!!! ({str(datetime.now())})')
    
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=str, default='0,1')
    p.add_argument('-v', '--verbose', type=str, default='0')
    p.add_argument('--stage', type=int, required=True)
    p.add_argument('--log_viterbi', type=bool, default=False)
    p.add_argument('--cpu_viterbi', type=bool, default=False)
    p.add_argument('--pre_trained_model', type=str, default='')
    args = p.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    
    if args.verbose=='0':
        import warnings
        warnings.filterwarnings("ignore")
        
    main(args)