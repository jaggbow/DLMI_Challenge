import torch
import os
import numpy as np
from PIL import Image
import glob
import pandas as pd


class LymphBags(torch.utils.data.Dataset):
    def __init__(self, bags_dir, df, mode='train', transforms=None):
        assert mode in [
            'train', 'test'], "mode must belong to ['train', 'test']"
        self.transforms = transforms
        self.mode = mode
        self.df = df
        self.dir = bags_dir
        self.bags = [i for i in df['ID']]

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
            gender = torch.as_tensor([self.df.loc[idx_][2]], dtype=torch.float)
            count = torch.as_tensor([self.df.loc[idx_][4]], dtype=torch.float)
            age = torch.as_tensor([self.df.loc[idx_][-1]], dtype=torch.float)
            label = self.df.loc[idx_][1]
            return images, gender, count, age, label
        else:
            bags = os.path.join(self.dir, self.bags[index])
            idx_ = self.df[self.df['ID'] == self.bags[index]].index[0]
            images = []
            for bag in os.listdir(bags):
                img = Image.open(os.path.join(bags, bag))
                if self.transforms:
                    img = self.transforms(img).unsqueeze(0)
                images.append(img)
            images = torch.cat(images)
            gender = torch.as_tensor([self.df.loc[idx_][2]], dtype=torch.float)
            count = torch.as_tensor([self.df.loc[idx_][4]], dtype=torch.float)
            age = torch.as_tensor([self.df.loc[idx_][-1]], dtype=torch.float)
            return images, gender, count, age, self.bags[index]


class LymphImages(torch.utils.data.Dataset):
    def __init__(self, train_path, transforms=None):

        self.transforms = transforms
        self.train_path = train_path

        self.L_img = glob.glob(os.path.join(self.train_path, '*/*.jpg'))

    def __len__(self):
        return len(self.L_img)

    def __getitem__(self, index):
        img = Image.open(self.L_img[index])
        return img


def metadata_build(path):
    df_train = pd.read_csv(path)
    df_train['GENDER'] = df_train.GENDER.apply(lambda x: int(x == 'F'))
    df_train['DOB'] = df_train['DOB'].apply(lambda x: x.replace("-", "/"))
    df_train['AGE'] = df_train['DOB'].apply(
        lambda x: 2020-int(x.split("/")[-1]))
    return df_train
