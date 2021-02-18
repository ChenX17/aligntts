'''
Date: 2021-01-21 13:04:34
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-18 00:55:34
'''
from text import symbols

################################
# Experiment Parameters        #
################################
seed=1234
n_gpus=1
output_directory = 'training_log'
# log_directory = 'aligntts_am_char_conv_rm_sos_eos_relu_blank0'
log_directory = 'debug'
data_path = 'data/LJSpeech-1.1/preprocessed'

training_files='filelists/ljs_audio_text_train_filelist.txt'
validation_files='filelists/ljs_audio_text_val_filelist.txt'
text_cleaners=['english_cleaners']


################################
# Audio Parameters             #
################################
sampling_rate=22050
filter_length=1024
hop_length=256
win_length=1024
n_mel_channels=80
mel_fmin=0
mel_fmax=8000.0

################################
# Model Parameters             #
################################
n_symbols=len(symbols)
# data_type='phone_seq' # 'phone_seq'
data_type='char_seq'
symbols_embedding_dim=256
mel_embedding_dim = 128
hidden_dim=256
dprenet_dim=256
postnet_dim=256
ff_dim=1024
n_heads=2
n_layers=6
max_db=2
min_db=-12

################################
# Optimization Hyperparameters #
################################
# lr=384**-0.5
lr=0.0001
warmup_steps=4000
grad_clip_thresh=1.0
batch_size=16
accumulation=1
iters_per_validation=100
iters_per_checkpoint=10000
train_steps = [300000, 40000, 80000, 10000]


