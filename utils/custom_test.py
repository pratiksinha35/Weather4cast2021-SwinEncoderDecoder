from torch.utils.data import Dataset
import h5py
import torch

class NWCSAFH5(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, 'r')
        
        self.data = self.h5_file['REFL-BT']  # shape: (240, 11, 252, 252)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        full_seq = self.data[idx]  # shape: (11, 252, 252)
        
        x = full_seq[:-1]  # first 10 timesteps
        y = full_seq[-1]   # predict the 11th timestep

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)      # shape: (10, 252, 252)
        y = torch.tensor(y, dtype=torch.float32)      # shape: (252, 252)

        metadata = {"index": idx}

        return x, y, metadata
