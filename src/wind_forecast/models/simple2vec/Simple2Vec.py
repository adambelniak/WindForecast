from torch import nn
import torch

class Simple2Vec(nn.Module):
    def __init__(self, num_features: int, embedding_size: int):
        super().__init__()
        self.simple2vec_dim = embedding_size
        self.wa = nn.Parameter(data=torch.empty(size=(1, num_features, self.simple2vec_dim)), requires_grad=True)
        self.ba = nn.Parameter(data=torch.empty(size=(1, num_features, self.simple2vec_dim)), requires_grad=True)

        self.wa.data.uniform_(-1, 1)
        self.ba.data.uniform_(-1, 1)

    def forward(self, inputs):
        dp = torch.mul(torch.unsqueeze(inputs, -1), self.wa) + self.ba
        return torch.reshape(dp, (-1, inputs.shape[1] * self.simple2vec_dim))
