import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from tqdm import trange, tqdm

from attacks.attack import Attack
from logger import CsvLogger
from run import save_checkpoint
from util.cross_entropy import CrossEntropyLoss


class Transfer(Attack):
    def __init__(self, net, loss, subst_net, subst_att):
        super(Transfer, self).__init__(net, loss)
        self.subst_net = subst_net
        self.att = subst_att(subst_net, loss)

    def generate_sample(self, x, y, eps, normalize):
        x_i, _, _ = self.att.generate_sample(x, y, eps, normalize)
        return x_i, self.net(x), self.net(x_i)
