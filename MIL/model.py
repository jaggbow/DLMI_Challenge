import torch
from torch import nn
from torch.functional import F


class Attention(nn.Module):
    def __init__(self, head, L=500, D=128, K=1):
        super(Attention, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor = head

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)  # NxL
        H = H.view(-1, 512)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.)

        return Y_prob, Y_hat, A
