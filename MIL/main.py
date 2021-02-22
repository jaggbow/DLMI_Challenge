import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime
import random
from tqdm import tqdm
from PIL import Image
# Sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
# Torch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# TorchVision
from torchvision import models, transforms, datasets
from torchvision.utils import make_grid

from data import LymphBags
from model import Attention
from trainer import Trainer

train_dir = 'input/trainset'
test_dir = 'input/testset'
device = 'cpu'

df_train = pd.read_csv('input/trainset/trainset_true.csv')
df_train['GENDER'] = df_train.GENDER.apply(lambda x: int(x == 'F'))
df_train['DOB'] = df_train['DOB'].apply(lambda x: x.replace("-", "/"))
df_train['AGE'] = df_train['DOB'].apply(lambda x: 2020-int(x.split("/")[-1]))
df_test = pd.read_csv('input/testset/testset_data.csv')

# Data Augmentation
tsfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
trainset = LymphBags(train_dir, df_train, transforms=tsfms)
testset = LymphBags(test_dir, df_test, transforms=tsfms)

train_loader = DataLoader(trainset, batch_size=1, shuffle=True)

# Define feature extractor
feat_extractor = models.resnet34(pretrained=True).to(device)
feat_extractor = torch.nn.Sequential(*(list(feat_extractor.children())[:-1]))
# Define model
model = Attention(head=feat_extractor, L=512)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3,
                       betas=(0.9, 0.999), weight_decay=1e-5)

# Initialize the Trainer
tr = Trainer(model=model, optimizer=optimizer,
             train_loader=train_loader, epochs=20, device=device)
tr.train()
