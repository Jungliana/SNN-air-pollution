import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class TwoLayerPerceptron(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.relu = nn.LeakyReLU()
        self.lin2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x


class LeakySNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif = snn.Leaky(beta=0.9, threshold=1., learn_beta=True,
                             learn_threshold=True, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        mem = self.lif.init_leaky()

        spk = self.fc1(x)
        for step in range(self.num_steps):
            spk, mem = self.lif(spk, mem)
        cur2 = self.fc2(spk)
        return cur2


class SynapticSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps=100):
        super().__init__()
        self.num_steps = num_steps
        self.spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif = snn.Synaptic(alpha=0.9, beta=0.85, threshold=1., learn_beta=True,
                                learn_threshold=True, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        syn, mem = self.lif.init_synaptic()

        spk = self.fc1(x)
        for step in range(self.num_steps):
            spk, syn, mem = self.lif(spk, syn, mem)
        cur2 = self.fc2(spk)
        return cur2


class DoubleLeakySNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.spike_grad = surrogate.fast_sigmoid()

        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.9, threshold=1., learn_beta=True,
                              learn_threshold=True, spike_grad=self.spike_grad)
        self.lif2 = snn.Leaky(beta=0.9, threshold=1., learn_beta=True,
                              learn_threshold=True, spike_grad=self.spike_grad)
        self.lin2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1 = self.lin1(x)
        for step in range(self.num_steps):
            spk2, mem1 = self.lif1(spk1, mem1)
            spk1, mem2 = self.lif2(spk2, mem2)
        cur2 = self.lin2(spk1)
        return cur2
