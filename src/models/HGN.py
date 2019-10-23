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
            layers = [c_block(1, o, s)]
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
        x1 = self.c_layer1(x)   # [3,64,64]    >>  [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>  [128,16,16]
        x3 = self.c_layer3(x2)  # [128,32,32]  >>  [256,8,8]
        x4 = self.c_layer4(x3)  # [256,32,32]  >>  [512,4,4]
        # x = self.c_layer5(x)

        """decoder"""
        # x = self.d_layer5(x)
        x5 = self.d_layer4(x4)+x3  # [512,4,4]    >>  [256,8,8]
        x6 = self.d_layer3(x5)+x2  # [256,8,8]    >>  [128,16,16]
        x7 = self.d_layer2(x6)  # [128,16,16]  >>  [64,32,32]
        y = self.d_layer1(x7)  # [64,32,32]   >>  [1,64,64]
        y = torch.transpose(y, 2, 3)

        """processing"""
        y = n(y)
        y = torch.stack((torch.cos(y*2*math.pi), torch.sin(y*2*math.pi)), dim=4)
        y = torch.fft(y, 2)
        y = torch.sqrt(y[:, :, :, :, 0].pow(2) + y[:, :, :, :, 1].pow(2))
        # y = n(y)
        return y
