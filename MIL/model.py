import torch
from torch import nn
from torch.functional import F


class Attention(nn.Module):
    def __init__(self, K=64):
        super(Attention, self).__init__()
        self.K = K

        self.feature_extractor = ResMIL(K=self.K)

        ## Attention #########################################
        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        # )

        ## Classifier ########################################
        self.classifier = nn.Sequential(
            nn.Linear(K*8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor(x)  # NxL
        H = H.view(-1, self.K*8)
        M = torch.mean(H, dim=0).unsqueeze(0)

        # A = self.attention(H)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_prob = torch.log(1e-10 + torch.cat([Y_prob, 1-Y_prob], dim=1))
        Y_hat = torch.argmax(Y_prob)
        return Y_prob, Y_hat


class ResMIL(nn.Module):
    def __init__(self, K=64):
        super(ResMIL, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, K, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(K),
            nn.ReLU(),
        )

        self.model = nn.Sequential(
            ResBlock(K, K),
            ResBlock(K, K*2),
            ResBlock(K*2, K*4),
            ResBlock(K*4, K*8, final=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, input):
        x = self.initial(input)
        x = self.model(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, K_in, K_out, final=False):
        super(ResBlock, self).__init__()

        self.K_in = K_in
        self.K_out = K_out
        self.final = final

        if self.final:
            self.l = nn.Sequential(
                nn.Conv2d(self.K_in, self.K_out, kernel_size=3,
                          stride=2, padding=1, bias=False),

                nn.BatchNorm2d(self.K_out),
                nn.ReLU(),

                nn.Conv2d(self.K_out, self.K_out, kernel_size=3,
                          stride=1, padding=1, bias=False),
            )
        else:
            self.l = nn.Sequential(
                nn.Conv2d(self.K_in, self.K_out, kernel_size=3,
                          stride=2, padding=1, bias=False),

                nn.BatchNorm2d(self.K_out),
                nn.ReLU(),

                nn.Conv2d(self.K_out, self.K_out, kernel_size=3,
                          stride=1, padding=1, bias=False),

                nn.BatchNorm2d(self.K_out),
                nn.ReLU(),
            )

        self.ds = nn.Sequential(
            nn.Conv2d(self.K_in, self.K_out,
                      kernel_size=(1, 1), stride=2, padding=0, bias=False),
            nn.BatchNorm2d(self.K_out)
        )

        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.l(input)
        ds_input = self.ds(input)
        return self.relu(ds_input + x)
