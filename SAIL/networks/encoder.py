import numpy as np
import torch.nn as nn
from SAIL.misc.utils import init


class Encoder(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_size, latent_size):
        super(Encoder, self).__init__()
        input_dim = ob_dim + ac_dim
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.z = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), nn.Tanh(),
                               init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                               init_(nn.Linear(hidden_size, latent_size)))
        self.train()

    def forward(self, inputs):
        x = inputs
        z = self.z(x)
        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_outputs):
        super(Decoder, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.pred_next_state = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), nn.Tanh(),
                                             init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
                                             init_(nn.Linear(hidden_size, num_outputs)))
        self.train()

    def forward(self, inputs):
        x = inputs
        pred_next_state = self.pred_next_state(x)
        return pred_next_state


class Enc_Dec_Net(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_size, latent_size):
        super(Enc_Dec_Net, self).__init__()
