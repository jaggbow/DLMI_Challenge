import torch
from tqdm import tqdm
import numpy as np
from torch.nn import NLLLoss, BCEWithLogitsLoss


class Trainer():

    def __init__(self, model, optimizer, train_loader, epochs, device):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.epochs = epochs
        self.device = device

        self.loss_f = BCEWithLogitsLoss()

    def train(self):

        self.model.train()

        for epoch in range(self.epochs):

            train_loss = []
            train_acc = []

            for batch_idx, (data, label) in enumerate(tqdm(self.train_loader)):
                bag_label = label[0].unsqueeze(0).unsqueeze(1)
                data, bag_label = data.to(
                    self.device), bag_label.to(self.device)

                # reset gradients
                self.optimizer.zero_grad()

                # calculate loss and metrics
                prediction, pred_hat, _ = self.model(data)
                loss = self.loss_f(prediction, bag_label.type_as(prediction))
                acc = (pred_hat == bag_label).float()

                train_loss.append(loss.item())
                train_acc.append(acc.item())

                # backward pass
                loss.backward()
                # step
                self.optimizer.step()

            # calculate loss and error for epoch
            mean_loss = np.mean(train_loss)
            mean_acc = np.mean(train_acc)

            print('Epoch: {}, Loss: {:.4f}, Train accuracy: {:.4f}'.format(
                epoch, mean_loss, mean_acc))
