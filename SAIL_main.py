import sys
import os
import time
import random
from collections import deque

import numpy as np
import torch

from SAIL.misc.utils import cleanup_log_dir
from SAIL.misc.arguments import get_args
from SAIL.networks.networks_manager import NetworksManager
from SAIL.rl.rl_agent import RLAgent


def setup(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    log_dir = os.path.expanduser(args.log_dir)
    cleanup_log_dir(log_dir)
    torch.set_num_threads(1)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")


def main():
    args = get_args()
    args.num_processes = 1
    print("== Starting SAIL with the following parameters ==")
    print(vars(args))
    setup(args)
    