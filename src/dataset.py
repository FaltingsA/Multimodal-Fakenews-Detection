import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import utils


class RumorDataset(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.types = torch.from_numpy(np.array(dataset['types']))
        print('TEXT: %d, Image: %d, label: %d, Mask: %d'
               % (len(self.text), len(self.image), len(self.label), len(self.mask)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.types[idx]

    def get_len(self):
        return len(self.text), len(self.image), len(self.label), len(self.mask)