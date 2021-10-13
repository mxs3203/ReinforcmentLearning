import numpy as np
import torch
import torch.nn
import torch.optim
from torch import nn
from torch.distributions import Categorical

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [nn.Linear(in_dim, 64),
                  nn.ReLU(),
                  nn.Linear(64, out_dim)]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        return self.model(x)

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))  # to tensor
        pdparam = self.forward(x)  # forward pass
        pd = Categorical(logits=pdparam)  # prob dist
        action = pd.sample()  # pi(a|s) in action
        log_prob = pd.log_prob(action)  # log_prob of pi
        self.log_probs.append(log_prob)  # store for training
        return action.item()

