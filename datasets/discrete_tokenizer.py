import numpy as np
import torch

class DiscreteTokenizer(object):
    def __init__(self, num_bins, seq_len):
        self.num_bins = num_bins
        vocab_size = num_bins * num_bins
        self.seq_len = seq_len
        self.vocab_size = vocab_size + 4
        # self.cls = vocab_size
        self.bos = vocab_size + 0
        self.eos = vocab_size + 1
        self.sep = vocab_size + 2
        self.pad = vocab_size + 3
        # self.vocab = {x: x for x in range(vocab_size)}
    
    def __len__(self):
        return self.vocab_size

    def __call__(self, seq, add_bos, add_eos, dtype):
        out = []
        if add_bos:
            out = [self.bos]
        for sub in seq:
            # out.append(self.cls) # cls token
            out.extend(sub)
            out.append(self.sep)
        out.pop(-1) # remove last separator token
        if self.seq_len > len(out):
            out.extend([self.pad] * (self.seq_len - len(out)))
        if add_eos:
            out[-1] = self.eos

        return torch.tensor(out, dtype=dtype)
    
    def _padding(self, seq, pad_value, dtype):
        if self.seq_len > len(seq):
            seq.extend([pad_value] * (self.seq_len - len(seq)))
        return torch.tensor(np.array(seq), dtype=dtype)