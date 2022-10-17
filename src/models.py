import torch.nn as nn
import snntorch as snn


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


class TwoLayerLeaky(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95, num_steps=25):
        super().__init__()
        self.num_steps = num_steps

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif.init_leaky()

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk, mem = self.lif(cur1, mem)
            cur2 = self.fc2(spk)
        return cur2


class ThreeLayerLeaky(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95, num_steps=25):
        super().__init__()
        self.num_steps = num_steps

        # Initialize layers
        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.relu = nn.LeakyReLU()
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lif = snn.Leaky(beta=beta)
        self.lin3 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif.init_leaky()

        for step in range(self.num_steps):
            cur1 = self.lin1(x)
            cur1 = self.bn1(cur1)
            cur1 = self.relu(cur1)
            cur2 = self.lin2(cur1)
            spk, mem = self.lif(cur2, mem)
            cur3 = self.lin3(spk)

        return cur3
