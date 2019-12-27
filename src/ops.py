from ops import *
import torch.nn as nn
import torch



class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        # Filter size of discriminator
        ndf = 64
        # Output image channels
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            r(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            r(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            r(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            r(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        output = self.main(img)
        return output.view(-1, 1).squeeze(1)


class LayerActivations():
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


""" Layers """


class c_block(nn.Module):
    def __init__(self, i, o, k, s, p):
        super(c_block, self).__init__()
        self.c1 = c(i, o, k=k, s=s, p=p)
        self.c2 = c(o, o, k=k, s=1, p=p)
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
    def __init__(self, i, o, k, s, p):
        super(d_block, self).__init__()
        self.d1 = d(i, o, k=k, s=s, p=p)
        self.d2 = d(o, o, k=k, s=1, p=p)
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

def t():
    return nn.Tanh()


""" Batch Normalization """


def b(channels):
    return nn.BatchNorm2d(channels)


"""Loss Functions"""


def sin_cos_loss(cos, sin):
    loss = (cos.pow(2) + sin.pow(2) - 1).pow(2)
    loss = loss.sum(0).sum(0).sum(0).sum(0)
    return loss/(64*64*64)


def minmaxLoss_onepoint(img):
    max1, _ = img.max(2)
    max2, _ = max1.max(2)

    min1, _ = img.min(2)
    min2, _ = min1.min(2)

    loss = max2-min2
    sum = loss.sum(0)

    return sum


def minmaxLoss_rowcolumn(img):

    for i in range(img.size()[0]):
        plane = img[i, :, :, :]
        meanval = plane.mean()
        plane = plane - meanval
        img[i, :, :, :] = abs(plane)

    sum = img.sum()

    return sum


def d1():
    return nn.Dropout()


def d2():
    return nn.Dropout2d()


def n1(tensor):
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


def n2(tensor):
    if torch.sub(torch.max(tensor), torch.min(tensor)) == 0:
        tensor = tensor / tensor
    else:
        tensor = torch.div(tensor, torch.max(tensor))
    return tensor
