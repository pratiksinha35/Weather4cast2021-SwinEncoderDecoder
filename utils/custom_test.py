from torch.utils.data import Dataset
import h5py
import torch
import torch.nn.functional as F

class NWCSAFH5(Dataset):
    def __init__(self, h5_path, seq_len=4):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        
        self.data = self.h5_file['REFL-BT']  # shape: (240, 11, 252, 252)
        self.seq_len = seq_len
        self.num_sequences = self.data.shape[0] // self.seq_len  # should be 60

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        
        # Shape: (4, 11, 252, 252)
        x = self.data[start:end]
        x = torch.tensor(x, dtype=torch.float32)

        # Padding each image in the sequence to (256, 256)
        # Current shape: (seq_len, channels, 252, 252)
        # Pad format: (left, right, top, bottom) => (0, 4, 0, 4)

        x = F.pad(x, pad=(0, 4, 0, 4), mode='constant', value=0)
        
        metadata = {"sequence_index": idx}

        return x, metadata
