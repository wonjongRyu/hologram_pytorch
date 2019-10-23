from ops import *
import torch.nn as nn
import torch

""" Layers """


class c_block(nn.Module):
    def __init__(self, i, o, s):
        super(c_block, self).__init__()
        self.c1 = c(i, o, k=3, s=s, p=1)
        self.c2 = c(o, o, k=3, s=1, p=1)
        self.b1 = b(o)
        self.b2 = b(o)
        self.r = r()
        self.shortcut = nn.Sequential()
        if s != 1 or i != o:
            self.shortcut = nn.Sequential(
                c(i, o, k=1, s=s, p=0),
                b(o)
            )

    def forward(self, x):
        y = self.r(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        y += self.shortcut(x)
        y = self.r(y)
        return y


class d_block(nn.Module):
    def __init__(self, i, o, s):
        super(d_block, self).__init__()
        self.d1 = d(i, o, k=3, s=s, p=1)
        self.d2 = d(o, o, k=3, s=1, p=1)
        self.b1 = b(o)
        self.b2 = b(o)
        self.r = r()
        self.shortcut = nn.Sequential()
        if s != 1 or i != o:
            self.shortcut = nn.Sequential(
                d(i, o, k=1, s=s, p=0),
                b(o)
            )

    def forward(self, x):
        y = self.r(self.b1(self.d1(x)))
        y = self.b2(self.d2(y))
        y += self.shortcut(x)
        y = self.r(y)
        return y


def c(i, o, k=1, s=1, p=1):
    return nn.Conv2d(in_channels=i, out_channels=o, kernel_size=k, stride=s, padding=p, bias=False)


def d(i, o, k=1, s=1, p=1):
    return nn.ConvTranspose2d(i, o, kernel_size=k, stride=s, padding=p, output_padding=s-1, bias=False)


def f(i, o):
    return nn.Linear(in_features=i, out_features=o)


def v1(i):
    dim = 1
    for s in i.size()[1:]:
        dim = dim * s
    return i.view(-1, dim)


def v2(i, size):
    return i.view(-1, size[0], size[1], size[2])


"""Activation Functions"""


def r():
    return nn.ReLU()


def l(alpha):
    return nn.LeakyReLU(alpha)


""" Batch Normalization """


def b(channels):
    return nn.BatchNorm2d(channels)


"""Loss Functions"""


def do(i):
    return nn.Dropout2d(i)


def n(tensor):
    if torch.sub(torch.max(tensor), torch.min(tensor)) == 0:
        tensor = tensor / tensor
    else:
        tensor = torch.div(
            torch.sub(
                tensor,
                torch.min(tensor)
            ),
            torch.sub(
                torch.max(tensor),
                torch.min(tensor)
            )
        )
    return tensor
