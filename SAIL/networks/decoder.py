import numpy as np
import torch.nn as nn
from SAIL.misc.utils import init


class Decoder(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(Decoder, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.pred_next_state = nn.Sequential(init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                                             init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                             init_(nn.Linear(hidden_size, num_outputs)))
        self.train()

    def forward(self, inputs):
        x = inputs
        pred_next_state = self.pred_next_state(x)
        return pred_next_state