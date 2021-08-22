import pandas as pd
import os, sys

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MNISTDataset(Dataset):
    label_tensor = None
    data_tensor = None

    def __init__(self, data_file, with_label):
        if with_label:
            self.label_tensor = torch.LongTensor(pd.read_csv(data_file).iloc[::, 0].values.reshape(-1))
            self.data_tensor = torch.FloatTensor(
                pd.read_csv(data_file).iloc[:, 1:].values.reshape(-1, 28, 28)).unsqueeze(1)
        else:
            self.data_tensor = torch.FloatTensor(
                pd.read_csv(data_file).iloc[:, 0:].values.reshape(-1, 28, 28)).unsqueeze(1)

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        data = self.data_tensor[idx]
        if self.label_tensor is None:
            label = torch.zeros(1)
        else:
            label = self.label_tensor[idx]

        return data, label
