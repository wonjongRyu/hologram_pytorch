from ops import *
import torch
import math


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class HGN(nn.Module):
    def __init__(self, block_num):
        super(HGN, self).__init__()
        self.o = 64

        """ Layers """
        # self.r = r()
        # self.b = b(self.o // 2)
        # self.c = c(i=3, o=self.o // 2, k=7, s=2, p=3)  # 16

        self.c_layer1 = self.make_c_layer(self.o, block_num[0], s=2)
        self.c_layer2 = self.make_c_layer(self.o, block_num[1], s=2)
        self.c_layer3 = self.make_c_layer(self.o, block_num[2], s=2)
        self.c_layer4 = self.make_c_layer(self.o, block_num[3], s=2)
        # self.c_layer5 = self.make_c_layer(self.o, block_num[4], s=2)

        self.o //= 4

        # self.d_layer5 = self.make_d_layer(self.o, block_num[4], s=2)
        self.d_layer4 = self.make_d_layer(self.o, block_num[3], s=2)
        self.d_layer3 = self.make_d_layer(self.o, block_num[2], s=2)
        self.d_layer2 = self.make_d_layer(self.o, block_num[1], s=2)
        self.d_layer1 = self.make_d_layer(1, block_num[0], s=2)

    def make_c_layer(self, o, num_blocks, s):
        if o == 64:
            layers = [c_block(3, o, s)]
            for _ in range(num_blocks - 1):
                layers.append(c_block(o, o, s=1))

        else:
            layers = [c_block(o // 2, o, s)]
            for _ in range(num_blocks - 1):
                layers.append(c_block(o, o, s=1))

        self.o = o * 2
        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, o, num_blocks, s):
        if o == 1:
            layers = [d_block(self.o * 2, 1, s)]
            for _ in range(num_blocks - 1):
                layers.append(d_block(1, 1, s=1))
        else:
            layers = [d_block(o * 2, o, s)]
            for _ in range(num_blocks - 1):
                layers.append(d_block(o, o, s=1))
            self.o = o // 2

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """
        input: [64,64,1] image
        output: [64,64,1] hologram
        """

        """ encoder """
        # x = self.r(self.b(self.c(x)))
        x = self.c_layer1(x)
        x = self.c_layer2(x)
        x = self.c_layer3(x)
        x = self.c_layer4(x)
        # x = self.c_layer5(x)

        """decoder"""
        # x = self.d_layer5(x)
        x = self.d_layer4(x)
        x = self.d_layer3(x)
        x = self.d_layer2(x)
        x = self.d_layer1(x)

        """processing"""
        # x = n(x)
        # x = torch.stack((torch.cos(x*2*math.pi), torch.sin(x*2*math.pi)), dim=4)
        # x = torch.fft(x, 2)
        # x = torch.sqrt(x[:, :, :, :, 0].pow(2) + x[:, :, :, :, 1].pow(2))
        # x = n(x)
        return x
