import numpy as np
import editdistance
import re
import os

from text import text_to_sequence, sequence_to_text_b

re_filter = re.compile(r"[^a-zA-Z ]")

seq_path = 'data/LJSpeech-1.1/preprocessed/decode_am_chars'

def compute_cer(results):
    """
    Arguments:
        results (list): list of ground truth and
        predicted sequence pairs.
    Returns the CER for the full set.
    """
    distance = sum(editdistance.eval(label, pred)
                for label, pred in results)
    total = sum(len(label) for label, _ in results)
    return float(distance)/float(total)
datasets = ['train', 'val', 'test']
for dataset in datasets:
    to_pro = []
    
    with open(f'filelists/ljs_audio_text_{dataset}_filelist.txt', 'r', encoding='utf-8') as f:
        lines_raw = [line.split('|') for line in f.read().splitlines()]
        for i in range(len(lines_raw)):
            file_name, _, text = lines_raw[i]
            res = sequence_to_text_b(np.load(os.path.join(seq_path, file_name+'_seq.npy'))).lower()
            if re_filter.search(res):
                # print(res)
                res = re_filter.sub('', res)
                #import pdb;pdb.set_trace()
            ref = re_filter.sub('', text).lower()
            to_pro.append((ref, res))
    cer = compute_cer(to_pro)
    print(cer)
    import pdb;pdb.set_trace()
    
