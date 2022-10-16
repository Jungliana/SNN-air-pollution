import snntorch as snn
import torch.nn as nn


class TwoLayerLeaky(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.95, num_steps=25):
        super().__init__()
        self.num_steps = num_steps

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        # Initialize hidden states at t=0
        mem = self.lif1.init_leaky()

        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk, mem = self.lif1(cur1, mem)
            cur2 = self.fc2(spk)
        return cur2
