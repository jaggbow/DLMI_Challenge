import torch
from tqdm import tqdm
import numpy as np
from torch.nn import NLLLoss, BCEWithLogitsLoss
from sklearn.metrics import balanced_accuracy_score


class Trainer():

    def __init__(self, model, optimizer, train_loader, epochs, device, valid_loader=None, save_path='saved_model.pt', batch_size=4):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.batch_size = batch_size

        self.loss_f = NLLLoss()

        self.logs = {
            'Loss': [],
            'Accuracy': [],
            'Balanced Acc': [],
            'Validation Loss': [],
            'Validation Accuracy': [],
            'Validation Balanced Acc': []
        }

    def train(self):

        for epoch in range(self.epochs):

            self.model.train()

            train_loss = []
            train_acc = []
            y_true, y_pred = [], []
            loss, ind_loss = 0, 0
            # reset gradients
            self.optimizer.zero_grad()

            for batch_idx, (data, label) in enumerate(self.train_loader):

                bag_label = label[0].unsqueeze(0)
                data, bag_label = data.to(
                    self.device), bag_label.to(self.device)

                # reset gradients
                self.optimizer.zero_grad()

                # calculate loss and metrics
                prediction, pred_hat = self.model(data)

                loss += self.loss_f(prediction, bag_label)
                ind_loss += 1

                acc = (pred_hat == bag_label).float()

                train_acc.append(acc.item())
                y_true.append(bag_label.item())
                y_pred.append(pred_hat.item())

                if ind_loss >= self.batch_size:
                    train_loss.append(loss.item()/ind_loss)
                    loss = loss/ind_loss
                    # backward pass
                    loss.backward()
                    # step
                    self.optimizer.step()
                    # reset gradients
                    self.optimizer.zero_grad()
                    loss, ind_loss = 0, 0

            # If the last batch is smaller than batchsize
            if ind_loss > 0:
                train_loss.append(loss.item()/ind_loss)
                loss = loss/ind_loss
                # backward pass
                loss.backward()
                # step
                self.optimizer.step()
                # reset gradients
                self.optimizer.zero_grad()
                loss, ind_loss = 0, 0

            # calculate loss and error for epoch
            mean_loss = np.mean(train_loss)
            mean_acc = np.mean(train_acc)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)

            print('Epoch: {:03d}, Loss: {:.4f}, Accuracy: {:.4f}, Balanced Acc: {:.4f}'.format(
                epoch, mean_loss, mean_acc, balanced_acc), end=' ')

            self.logs['Loss'].append(mean_loss)
            self.logs['Accuracy'].append(mean_acc)
            self.logs['Balanced Acc'].append(balanced_acc)

            if self.valid_loader:
                self.evaluate()

    def evaluate(self):

        self.model.eval()

        test_loss = []
        test_acc = []
        y_true, y_pred = [], []

        for batch_idx, (data, label) in enumerate(self.valid_loader):

            with torch.no_grad():

                bag_label = label[0].unsqueeze(0)
                data, bag_label = data.to(
                    self.device), bag_label.to(self.device)

                prediction, pred_hat = self.model(data)
                loss = self.loss_f(prediction, bag_label)
                acc = (pred_hat == bag_label).float()

                test_loss.append(loss.item())
                test_acc.append(acc.item())
                y_true.append(bag_label.item())
                y_pred.append(pred_hat.item())

        # calculate loss and error for epoch
        mean_loss = np.mean(test_loss)
        mean_acc = np.mean(test_acc)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        print('Validation Loss: {:.4f}, Validation accuracy: {:.4f}, Validation Balanced Acc: {:.4f}'.format(
            mean_loss, mean_acc, balanced_acc))

        self.logs['Validation Loss'].append(mean_loss)
        self.logs['Validation Accuracy'].append(mean_acc)
        self.logs['Validation Balanced Acc'].append(balanced_acc)

        if max([0.]+self.logs['Validation Balanced Acc'][:-1]) < balanced_acc:
            print('Saving model ...')
            self.save_model()

    def predict(self, test_loader):

        self.model.eval()
        sub_dict = {"ID": [], "Predicted": []}
        L = []

        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(tqdm(test_loader)):
                data = data.to(self.device)
                prediction, pred_hat = self.model(data)
                sub_dict['Predicted'].append(pred_hat.item())
                sub_dict['ID'].append(label[0])

        return sub_dict

    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)

    def restore(self, model_path):
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
