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
mode = 'train'


if mode == 'train':

    # Metadatas
    df_train = metadata_build(train_meta)
    df_train, df_valid, _, _, = train_test_split(
        df_train, df_train['LABEL'], stratify=df_train['LABEL'], random_state=42, test_size=0.2)
    df_test = metadata_build(test_meta)

    # Data Augmentation
    tt = transforms.RandomChoice([
        transforms.RandomRotation(0, 0),
        transforms.RandomRotation(90, 90),
        transforms.RandomRotation(180, 180),
        transforms.RandomRotation(270, 270)
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
    testset = LymphBags(test_dir, df_test, transforms=tsfms)

    dd = LymphImages(train_dir, tsfms)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=1, shuffle=True)

    # Define feature extractor
    feat_extractor = models.resnet34(pretrained=True).to(device)
    feat_extractor = torch.nn.Sequential(
        *(list(feat_extractor.children())[:-2]))
    # Define model
    model = Attention(head=feat_extractor, L=512)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           betas=(0.9, 0.999), weight_decay=1e-5)

    # Initialize the Trainer
    tr = Trainer(model=model, optimizer=optimizer,
                 train_loader=train_loader, epochs=20, device=device, valid_loader=valid_loader)
    tr.train()

else:

    test_dir = 'input/testset'
    test_meta = 'input/testset/testset_data.csv'
    device = 'cpu'

    # Metadatas
    df_test = metadata_build(test_meta)

    # Data Augmentation
    tsfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Datasets
    testset = LymphBags(test_dir, df_test, transforms=tsfms)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    # Define feature extractor
    feat_extractor = models.resnet34(pretrained=True).to(device)
    feat_extractor = torch.nn.Sequential(
        *(list(feat_extractor.children())[:-2]))

    # Define model
    model = Attention(head=feat_extractor, L=512)
    model = model.to(device)
    model.load_state_dict(torch.load(
        'saved_model.pt', map_location=torch.device('cpu')))

    model.eval()
    L = []
    for batch_idx, (data, label) in enumerate(tqdm(test_loader)):
        bag_label = label[0].unsqueeze(0).unsqueeze(1)
        data, bag_label = data.to(
            device), bag_label.to(device)

        prediction, pred_hat, _ = model(data)
        L.append(prediction)

    print(L)
