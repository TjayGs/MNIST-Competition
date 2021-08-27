import os
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MNISTDataset(Dataset):
    label_tensor = None
    data_tensor = None
    transform = None

    def __init__(self, data_file, with_label, transform=None):
        """
        Will create a dataset for the given parameters. It is especially created for MNIST Dataset
        so there are some preconditions according the size of data for example
        :param data_file: the data file to read (with panda)
        :param with_label: it could be the case, that there are no label provided (test data, because of kaggle)
        :param transform: the transformations which should be applied on the fly
        """
        self.transform = transform
        if with_label:
            self.label_tensor = torch.LongTensor(pd.read_csv(data_file).iloc[::, 0].values.reshape(-1))
            self.data_tensor = torch.FloatTensor(
                pd.read_csv(data_file).iloc[:, 1:].values.reshape(-1, 28, 28)).unsqueeze(1)
        else:
            self.data_tensor = torch.FloatTensor(
                pd.read_csv(data_file).iloc[:, 0:].values.reshape(-1, 28, 28)).unsqueeze(1)

    def __len__(self):
        return len(self.data_tensor)

    def show_img(self, tensor_img):
        """
        This method could help you to show the image of one image tensor with the shape
        {C, W, H}
        """
        if random.randint(0, 1) == 0:
            plt.imshow(tensor_img.squeeze(axis=0).numpy(), cmap='gray')

    def __getitem__(self, idx):
        data = self.data_tensor[idx]
        if self.label_tensor is None:
            label = torch.zeros(1)
        else:
            label = self.label_tensor[idx]

        if self.transform:
            data = self.transform(data)

        return data, label
