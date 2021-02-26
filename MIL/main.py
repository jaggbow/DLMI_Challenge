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

from data import LymphBags, LymphImages, metadata_build
from model import Attention
from trainer import Trainer

train_dir = 'input/trainset'
train_meta = 'input/trainset/trainset_true.csv'
test_dir = 'input/testset'
test_meta = 'input/testset/testset_data.csv'
device = 'cpu'
batch_size = 4
model_path = 'saved_model1.pt'
mode = 'train'

# Metadatas
df_train = metadata_build(train_meta)
df_train, df_valid, _, _, = train_test_split(
    df_train, df_train['LABEL'], stratify=df_train['LABEL'], random_state=42, test_size=0.2)
df_test = metadata_build(test_meta)

# Data Augmentation
tt = transforms.RandomChoice([
    transforms.RandomRotation((0, 0)),
    transforms.RandomRotation((90, 90)),
    transforms.RandomRotation((180, 180)),
    transforms.RandomRotation((270, 270))
]
)

tsfms = transforms.Compose([
    tt,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Datasets
trainset = LymphBags(train_dir, df_train, transforms=tsfms)
validset = LymphBags(train_dir, df_valid, transforms=tsfms)
testset = LymphBags(test_dir, df_test, transforms=tsfms, mode='test')

dd = LymphImages(train_dir, tsfms)
train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=True)
valid_loader = DataLoader(validset, batch_size=1, shuffle=True)

# Define model
model = Attention()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4,
                       betas=(0.9, 0.999), weight_decay=1e-5)

# Initialize the Trainer
tr = Trainer(model=model, optimizer=optimizer,
             train_loader=train_loader, epochs=20, device=device, valid_loader=valid_loader, save_path=model_path, batch_size=batch_size)

if mode == 'train':
    # tr.restore(model_path)
    tr.train()

else:
    # tr.restore(model_path)
    dict = tr.predict(test_loader)
