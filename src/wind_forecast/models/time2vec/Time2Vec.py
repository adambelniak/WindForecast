from torch import nn
import torch

class Time2Vec(nn.Module):
    def __init__(self, num_features: int, embedding_size: int):
        super().__init__()
        self.time2vec_dim = embedding_size - 1
        # trend
        self.wb = nn.Parameter(data=torch.empty(size=(num_features,)), requires_grad=True)
        self.bb = nn.Parameter(data=torch.empty(size=(num_features,)), requires_grad=True)

        # periodic
        self.wa = nn.Parameter(data=torch.empty(size=(1, num_features, self.time2vec_dim)), requires_grad=True)
        self.ba = nn.Parameter(data=torch.empty(size=(1, num_features, self.time2vec_dim)), requires_grad=True)

        self.wb.data.uniform_(-1, 1)
        self.bb.data.uniform_(-1, 1)
        self.wa.data.uniform_(-1, 1)
        self.ba.data.uniform_(-1, 1)

    def forward(self, inputs):
        bias = torch.mul(self.wb, inputs) + self.bb
        dp = torch.mul(torch.unsqueeze(inputs, -1), self.wa) + self.ba
        wgts = torch.sin(dp)

        ret = torch.cat([torch.unsqueeze(bias, -1), wgts], -1)
        ret = torch.reshape(ret, (-1, inputs.shape[1] * (self.time2vec_dim + 1)))
        return ret