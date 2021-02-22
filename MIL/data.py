import torch
import os
import numpy as np
from PIL import Image


class LymphBags(torch.utils.data.Dataset):
    def __init__(self, bags_dir, df, mode='train', transforms=None):
        assert mode in [
            'train', 'test'], "mode must belong to ['train', 'test']"
        self.transforms = transforms
        self.mode = mode
        self.df = df
        self.dir = bags_dir
        self.bags = list(filter(lambda x: x[0] == 'P', os.listdir(bags_dir)))

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        if self.mode == 'train':
            bags = os.path.join(self.dir, self.bags[index])
            images = []
            for bag in os.listdir(bags):
                img = Image.open(os.path.join(bags, bag))
                if self.transforms:
                    img = self.transforms(img).unsqueeze(0)
                images.append(img)
            images = torch.cat(images)
            idx_ = self.df[self.df['ID'] == self.bags[index]].index[0]
            label = self.df.iloc[idx_, 1]
            return images, label
        else:
            bags = os.path.join(self.dir, self.bags[index])
            images = []
            for bag in os.listdir(bags):
                img = Image.open(os.path.join(bags, bag))
                if self.transforms:
                    img = self.transforms(img).unsqueeze(0)
                images.append(img)
            images = torch.cat(images)
            return images
