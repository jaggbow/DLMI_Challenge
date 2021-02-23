import torch
from tqdm import tqdm
import numpy as np
from torch.nn import NLLLoss, BCEWithLogitsLoss


class Trainer():

    def __init__(self, model, optimizer, train_loader, epochs, device, valid_loader=None):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.device = device

        self.loss_f = NLLLoss()

    def train(self):

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = []
            train_acc = []

            for batch_idx, (data, label) in enumerate(self.train_loader):
                bag_label = label[0].unsqueeze(0)
                data, bag_label = data.to(
                    self.device), bag_label.to(self.device)

                # reset gradients
                self.optimizer.zero_grad()

                # calculate loss and metrics
                prediction, pred_hat = self.model(data)
                loss = self.loss_f(prediction, bag_label)
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

            print('Epoch: {:03d}, Loss: {:.4f}, Train accuracy: {:.4f}'.format(
                epoch, mean_loss, mean_acc), end=' ')

            if self.valid_loader:
                self.evaluate()

    def evaluate(self):

        self.model.eval()

        test_loss = []
        test_acc = []

        for batch_idx, (data, label) in enumerate(self.valid_loader):
            bag_label = label[0].unsqueeze(0)
            data, bag_label = data.to(
                self.device), bag_label.to(self.device)

            prediction, pred_hat, _ = self.model(data)
            loss = self.loss_f(prediction, bag_label)
            acc = (pred_hat == bag_label).float()

            test_loss.append(loss.item())
            test_acc.append(acc.item())

        # calculate loss and error for epoch
        mean_loss = np.mean(test_loss)
        mean_acc = np.mean(test_acc)

        print('Validation Loss: {:.4f}, Validation accuracy: {:.4f}'.format(
            mean_loss, mean_acc))

    def predict(self, test_loader):

        self.model.eval()
        L = []

        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0].unsqueeze(0).unsqueeze(1)
            data, bag_label = data.to(
                self.device), bag_label.to(self.device)

            prediction, pred_hat, _ = self.model(data)
            L.append(prediction)

        return L
