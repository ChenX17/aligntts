'''
Date: 2021-01-30 11:29:02
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2021-02-01 13:09:40
'''
import numpy as np
from text import sequence_to_text_b

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.tile(np.expand_dims(np.sum(np.exp(x), axis=1), axis=1), (1,x.shape[1])) 

probs = np.load('probs_array.npy')
logits = np.load('logits_array.npy')
import pdb;pdb.set_trace()
# probs = softmax(probs[:,0,:])
seq = np.argmax(probs[:, 0, :], axis=-1)
logits_seq = np.argmax(logits[0, :, :], axis=-1)

for i in range(seq.shape[-1]):
    text_seq = sequence_to_text_b(seq)
    print(text_seq)
    import pdb;pdb.set_trace()