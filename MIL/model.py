import torch
from torch import nn
from torch.functional import F


class Attention(nn.Module):
    def __init__(self, K=64, L=128, dropout_rate=0.2, conv_bias=True):
        super(Attention, self).__init__()

        self.K = K
        self.L = L
        self.conv_bias = True
        self.dropout_rate = dropout_rate
        self.conv_bias = conv_bias

        self.feature_extractor = ResMIL(
            K=self.K, dropout_rate=self.dropout_rate, conv_bias=self.conv_bias)

        # Attention #########################################
        self.attention = nn.Sequential(
            nn.Linear(self.K*8, self.L),
            nn.Tanh(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.L, 1)
        )

        ## Classifier ########################################
        self.classifier = nn.Sequential(
            nn.Linear(K*8 + 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x,  gender, count, age):
        x = x.squeeze(0)

        H = self.feature_extractor(x)  # NxL
        H = H.view(-1, self.K*8)

        # Attention
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL

        M = torch.cat([M, gender, age, count], dim=1)

        Y_prob = self.classifier(M)
        Y_prob = torch.log(1e-10 + torch.cat([Y_prob, 1-Y_prob], dim=1))
        Y_hat = torch.argmax(Y_prob)
        return Y_prob, Y_hat


class ResMIL(nn.Module):
    def __init__(self, K=64, dropout_rate=0.2, conv_bias=True):
        super(ResMIL, self).__init__()

        self.conv_bias = conv_bias
        self.dropout_rate = dropout_rate
        self.K = K

        self.initial = nn.Sequential(
            nn.Conv2d(3, self.K, kernel_size=7, stride=2,
                      padding=3, bias=self.conv_bias),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        self.model = nn.Sequential(
            ResBlock(self.K, self.K, dropout_rate=dropout_rate,
                     conv_bias=self.conv_bias),
            ResBlock(self.K, self.K*2, dropout_rate=dropout_rate,
                     conv_bias=self.conv_bias),
            ResBlock(self.K*2, self.K*4, dropout_rate=dropout_rate,
                     conv_bias=self.conv_bias),
            ResBlock(self.K*4, self.K*8, final=True,
                     dropout_rate=dropout_rate, conv_bias=self.conv_bias),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, input):
        x = self.initial(input)
        x = self.model(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, K_in, K_out, final=False, dropout_rate=0.2, conv_bias=True):
        super(ResBlock, self).__init__()

        self.K_in = K_in
        self.K_out = K_out
        self.final = final
        self.dropout_rate = dropout_rate
        self.conv_bias = conv_bias

        if self.final:
            self.l = nn.Sequential(
                nn.Conv2d(self.K_in, self.K_out, kernel_size=3,
                          stride=2, padding=1, bias=self.conv_bias),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),

                nn.Conv2d(self.K_out, self.K_out, kernel_size=3,
                          stride=1, padding=1, bias=self.conv_bias),
            )
        else:
            self.l = nn.Sequential(
                nn.Conv2d(self.K_in, self.K_out, kernel_size=3,
                          stride=2, padding=1, bias=self.conv_bias),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),

                nn.Conv2d(self.K_out, self.K_out, kernel_size=3,
                          stride=1, padding=1, bias=self.conv_bias),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
            )

        self.ds = nn.Sequential(
            nn.Conv2d(self.K_in, self.K_out,
                      kernel_size=(1, 1), stride=2, padding=0, bias=self.conv_bias),
        )

        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.l(input)
        ds_input = self.ds(input)
        return self.relu(ds_input + x)
