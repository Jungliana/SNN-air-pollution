from torch import full, float32
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class MultiLayerANN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs=1):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.sigmoid = nn.Sigmoid()
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.relu = nn.LeakyReLU()
        self.lin3 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.lin1(x)
        x = self.sigmoid(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x


class LeakySNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs=1, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.spike_grad = surrogate.fast_sigmoid()
        self.thr = full([num_hidden], 1., dtype=float32)

        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.lif = snn.Leaky(beta=0.9, threshold=self.thr, learn_beta=True,
                             learn_threshold=True, spike_grad=self.spike_grad)
        self.lin2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        mem = self.lif.init_leaky()

        x1 = self.lin1(x)
        spk_sum, mem = self.lif(x1, mem)
        for step in range(self.num_steps-1):
            spk, mem = self.lif(x1, mem)
            spk_sum += spk
        x2 = self.lin2(spk_sum)
        return x2


class SynapticSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs=1, num_steps=50):
        super().__init__()
        self.num_steps = num_steps
        self.spike_grad = surrogate.fast_sigmoid()
        self.thr = full([num_hidden], 1., dtype=float32)

        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.lif = snn.Synaptic(alpha=0.9, beta=0.85, threshold=self.thr, learn_beta=True,
                                learn_threshold=True, spike_grad=self.spike_grad)
        self.lin2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        syn, mem = self.lif.init_synaptic()

        x1 = self.lin1(x)
        spk_sum, syn, mem = self.lif(x1, syn, mem)
        for step in range(self.num_steps-1):
            spk, syn, mem = self.lif(x1, syn, mem)
            spk_sum += spk
        x2 = self.lin2(spk_sum)
        return x2


class DoubleLeakySNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs=1, num_steps=25):
        super().__init__()
        self.num_steps = num_steps
        self.spike_grad = surrogate.fast_sigmoid()
        self.thr1 = full([num_hidden], 1., dtype=float32)
        self.thr2 = full([num_hidden], 1., dtype=float32)

        self.lin1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.9, threshold=self.thr1, learn_beta=True,
                              learn_threshold=True, spike_grad=self.spike_grad)
        self.lif2 = snn.Leaky(beta=0.9, threshold=self.thr2, learn_beta=True,
                              learn_threshold=True, spike_grad=self.spike_grad)
        self.lin2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk1 = self.lin1(x)
        for step in range(self.num_steps):
            spk2, mem1 = self.lif1(spk1, mem1)
            spk1, mem2 = self.lif2(spk2, mem2)
        x2 = self.lin2(spk1)
        return x2
