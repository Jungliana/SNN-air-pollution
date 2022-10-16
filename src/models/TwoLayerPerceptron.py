import torch.nn as nn


class TwoLayerPerceptron(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.relu = nn.LeakyReLU()
        self.lin2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x
