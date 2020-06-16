import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# from torch.nn.utils.rnn import pack_padded_sequence
from scipy.io import loadmat
import numpy as np 
import copy
from functools import partial

class XRMBData(Dataset):
    '''customized dataset object for XRMB '''
    def __init__(self, file_path='../data/XRMB_SEQ.mat', noisy=0, train=True, seq_len=1000):
        self.train = train
        self.seq_len = seq_len
        self.noisy = noisy
        data_dict = loadmat(file_path)
        self.lens = data_dict['LENGTHS'].astype(int).squeeze()
        cutoff = int(sum(self.lens[:1837]))
        idx = range(cutoff) if train else range(cutoff, 2430668)

        self.v1 = data_dict['MFCCS'][idx].astype(np.float32)
        self.v2 = data_dict['ARTICS'][idx].astype(np.float32)
        self.label = data_dict['LABELS'][idx].astype(int)
        self.split_sequence()

    def split_sequence(self):
        '''
        split raw inputs to sequences

        returns:
            sequeces: list of tuples ((seq_len*num_feat_v1, seq_len*num_feat_v2, seq_len))
        '''
        self.sequences = []
        start_idx = 0
        cutoff = np.arange(self.seq_len, len(self.label)+1, self.seq_len)

        if self.noisy == 0:
            for end_idx in cutoff:
                self.sequences.append((self.v1[start_idx:end_idx], self.v2[start_idx:end_idx], self.label[start_idx:end_idx]))
                start_idx = end_idx
        else:
            np.random.seed(1024)
            noisy_idx = copy.deepcopy(cutoff)
            noisy_idx[-1] = 0
            np.random.shuffle(noisy_idx)
            for end_idx, cur_nis in zip(cutoff, noisy_idx):
                noisy_val = v1[cur_nis: cur_nis + self.seq_len]
                self.sequences.append((self.v1[start_idx:end_idx] + self.noisy*noisy_val, self.v2[start_idx:end_idx], self.label[start_idx:end_idx]))
                start_idx = end_idx
                

    def __getitem__(self, idx):
        v1, v2, label = self.sequences[idx]
        return (torch.FloatTensor(v1),
                torch.FloatTensor(v2),
                torch.IntTensor(label).squeeze())
    
    def __len__(self):
        return len(self.sequences)

def collect_xrmb(batch, pad=True, blank=40):
    """
    args:
        batch - list of tuples (v1, v2, label) extraced from getitem()
    returns:
        v1, v2 - seq_len * batch_sise * num_feat
    """
    v1 = []
    v2 = []
    target = []
    v_len = []
    for (x1, x2, y) in batch:
        v1.append(x1)
        v2.append(x2)
        target.append(y)
        v_len.append(x1.shape[0])

    if pad:
        # v1,v2 - list of variable length tensors
        # [(seq_len1, num_feat), (seq_len2, num_feat), ...] ==> tensor(max_seq_len, batch_size, num_feat)
        # target - list of tensors [(seq_len,), (seq_len,), ...]
        v1 = pad_sequence(v1)
        v2 = pad_sequence(v2)
        v_len = torch.IntTensor(v_len)
        target, target_len = collapse_str(target, blank=blank)
    else:
        v1 = torch.cat(v1, dim=0)
        v2 = torch.cat(v2, dim=0)
        v_len = torch.IntTensor(v_len)
        target, target_len = collapse_str(target, blank=blank)
    return v1, v2, target, v_len, target_len

def collapse_str(strs, blank=40):
    """collapse repeat labels
    args: 
        strs - list of tensors [(seq_len,), (seq_len,), ...]
    returns:
        target - tensor(batch_size, max_seq_len)
    """
    target = []
    target_len = []
    for seq in strs:
        collapsed_target = [seq[0]]
        for label in seq:
            if label != collapsed_target[-1]:
                collapsed_target.append(label)
        collapsed_target = [i for i in collapsed_target if i != blank]
        target.append(torch.IntTensor(collapsed_target))
        target_len.append(len(collapsed_target))
    # set batch_first=true to obtain (batch_size, max_seq_len) instead of (max_seq_len, batch_size)
    target = pad_sequence(target, padding_value=0, batch_first=True)
    target_len = torch.IntTensor(target_len)
    return target, target_len


class XRMBDataloader(DataLoader):
    def __init__(self, *args, pad=True, blank=40, **kwargs):
        super(XRMBDataloader, self).__init__(*args, **kwargs)
        self.collate_fn = partial(collect_xrmb, pad=pad, blank=blank)

def get_loader(batch, blank, noisy, seq_len=1000):
    train_data = XRMBData(noisy=noisy, seq_len=seq_len)
    test_data = XRMBData(train=False, noisy=noisy, seq_len=seq_len)
    train_loader = XRMBDataloader(dataset=train_data, batch_size=batch, shuffle=True, blank=blank)
    test_loader = XRMBDataloader(dataset=test_data, batch_size=batch, shuffle=False, blank=blank)
    return train_loader, test_loader

if __name__ == "__main__":
    mfcc = XRMBData(train=False, seq_len=1000)
    dl = XRMBDataloader(dataset=mfcc, batch_size=3, pad=True)
    v1, v2, t, vl, tl = next(iter(dl))
    print(v1.shape)
    print(t)
    print(vl)
    print(tl)
