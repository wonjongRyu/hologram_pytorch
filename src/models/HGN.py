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


class HGN_size(nn.Module):
    def __init__(self, block_num):
        super(HGN_size, self).__init__()

        """ Layers """

        self.c_layer1 = self.make_c_layer(1, 64, block_num[0], s=2)
        self.c_layer2 = self.make_c_layer(64, 128, block_num[1], s=2)
        self.c_layer3 = self.make_c_layer(128, 256, block_num[2], s=2)
        self.c_layer4 = self.make_c_layer(256, 512, block_num[3], s=2)
        self.c_layer5 = self.make_c_layer(512, 1024, block_num[4], s=2)
        self.d_layer5 = self.make_d_layer(1024, 512, block_num[4], s=2)
        self.d_layer4 = self.make_d_layer(512, 256, block_num[3], s=2)
        self.d_layer3 = self.make_d_layer(256, 128, block_num[2], s=2)
        self.d_layer2 = self.make_d_layer(128, 64, block_num[1], s=2)
        self.d_layer1 = self.make_d_layer(64, 1, block_num[0], s=2)

        self.c_layer11 = self.make_c_layer(1, 128, block_num[0], s=2)
        self.c_layer21 = self.make_c_layer(128, 256, block_num[1], s=2)
        self.c_layer31 = self.make_c_layer(256, 512, block_num[2], s=2)
        self.c_layer41 = self.make_c_layer(512, 1024, block_num[3], s=2)

        self.c_layer12 = self.make_c_layer(1, 256, block_num[0], s=2)
        self.c_layer22 = self.make_c_layer(256, 512, block_num[1], s=2)
        self.c_layer32 = self.make_c_layer(512, 1024, block_num[2], s=2)


        self.f1 = f(4096, 4096)
        self.f2 = f(4096, 4096)

        self.r = r()
        self.t = t()

        self.m = nn.Upsample(scale_factor=0.5, mode='bilinear')

    def make_c_layer(self, i, o, num_blocks, s, k=3, p=1):
        layers = [c_block(i, o, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(c_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, i, o, num_blocks, s, k=3, p=1):
        layers = [d_block(i, o, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(d_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""
        xx = self.m(x)
        xxx = self.m(xx)
        h1 = x
        h2 = torch.stack((h1, torch.zeros(h1.size()).cuda()), dim=4)
        h3 = torch.ifft(h2, 2)
        h4 = torch.atan2(h3[:, :, :, :, 1], h3[:, :, :, :, 0])

        """ encoder """
        x1 = self.c_layer1(x)   # [1,64,64]    >>   [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>   [128,16,16]
        x3 = self.c_layer3(x2)  # [128,16,16]  >>   [256,8,8]
        x4 = self.c_layer4(x3)  # [256,8,8]    >>   [512,4,4]
        x5 = self.c_layer5(x4)  # [512,4,4]    >>   [1024,2,2]

        xx1 = self.c_layer11(xx)  # [1,32,32]      >>    [128,16,16]
        xx2 = self.c_layer21(xx1) # [128,64,64]    >>    [256,8,8]
        xx3 = self.c_layer31(xx2) # [256,64,64]    >>    [512,4,4]
        xx4 = self.c_layer41(xx3) # [512,64,64]    >>    [1024,2,2]

        xxx1 = self.c_layer12(xxx)  # [1,16,16]    >>    [256,8,8]
        xxx2 = self.c_layer22(xxx1) # [256,8,8]    >>    [512,4,4]
        xxx3 = self.c_layer32(xxx2) # [512,4,4]    >>    [1024,2,2]

        """ fc layer """
        y1 = x5 + xx4 + xxx3
        y2 = v1(y1)
        y3 = self.r(self.f1(y2))
        y4 = v2(y3, [1024, 2, 2])

        """decoder1"""
        z6 = y4
        z4 = self.d_layer5(z6)  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4)  # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3)  # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2)  # [128,8,8]    >>  [64,16,16]
        z0 = self.d_layer1(z1)  # [64,16,16]  >>  [1,32,32]

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.t(self.f2(w2))
        w4 = v2(w3, [1, 64, 64])
        w5 = h4

        p1 = w5
        # p2 = torch.stack((torch.cos(p1 * math.pi), torch.sin(p1 * math.pi)), dim=4)
        p2 = torch.stack((torch.cos(p1), torch.sin(p1)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        p5 = n2(p4)

        return p1, p5

class HGN128(nn.Module):
    def __init__(self, block_num):
        super(HGN128, self).__init__()

        """ Layers """
        self.c = c(1, 32, k=3, s=2, p=1)
        self.b = b(32)
        self.c_layer1 = self.make_c_layer(32, 64, block_num[0], s=2)
        self.c_layer2 = self.make_c_layer(64, 128, block_num[1], s=2)
        self.c_layer3 = self.make_c_layer(128, 256, block_num[2], s=2)
        self.c_layer4 = self.make_c_layer(256, 512, block_num[3], s=2)
        self.c_layer5 = self.make_c_layer(512, 1024, block_num[4], s=2)
        self.d_layer5 = self.make_d_layer(1024, 512, block_num[4], s=2)
        self.d_layer4 = self.make_d_layer(512, 256, block_num[3], s=2)
        self.d_layer3 = self.make_d_layer(256, 128, block_num[2], s=2)
        self.d_layer2 = self.make_d_layer(128, 64, block_num[1], s=2)
        self.d_layer1 = self.make_d_layer(64, 1, block_num[0], s=2)

        self.f1 = f(4096, 4096)
        self.f2 = f(4096, 16384)

        self.r = r()
        self.t = t()

    def make_c_layer(self, i, o, num_blocks, s, k=3, p=1):
        layers = [c_block(i, o, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(c_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, i, o, num_blocks, s, k=3, p=1):
        layers = [d_block(i, o, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(d_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net


    def forward(self, x):
        """ forward function"""
        x0 = self.c(x)
        x0 = self.b(x0)
        x0 = self.r(x0)

        """ encoder """
        x1 = self.c_layer1(x0)  # [1,64,64]    >>  [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>  [128,16,16]
        x3 = self.c_layer3(x2)  # [128,16,16]  >>  [256,8,8]
        x4 = self.c_layer4(x3)  # [256,8,8]  >>  [512,4,4]
        x5 = self.c_layer5(x4)  # [512,4,4]  >>  [1024,2,2]

        """ fc layer """
        y1 = x5
        y2 = v1(y1)
        y3 = self.r(self.f1(y2))
        y4 = v2(y3, [1024, 2, 2])

        """decoder1"""
        z6 = y4
        z4 = self.d_layer5(z6) + x4  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4) + x3 # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3) + x2 # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2)  # [128,16,16]    >>  [64,32,32]
        z0 = self.d_layer1(z1)  # [64,32,32]  >>  [1,64,64]

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.t(self.f2(w2))
        w4 = v2(w3, [1, 128, 128])

        p1 = w4
        p2 = torch.stack((torch.cos(p1 * math.pi), torch.sin(p1 * math.pi)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        p5 = n2(p4)

        return p1, p5

class HGN(nn.Module):
    def __init__(self, block_num):
        super(HGN, self).__init__()

        """ Layers """
        self.c_layer1 = self.make_c_layer(1, 64, block_num[0], s=2)
        self.c_layer2 = self.make_c_layer(64, 128, block_num[1], s=2)
        self.c_layer3 = self.make_c_layer(128, 256, block_num[2], s=2)
        self.c_layer4 = self.make_c_layer(256, 512, block_num[3], s=2)
        self.c_layer5 = self.make_c_layer(512, 1024, block_num[4], s=2)
        self.d_layer5 = self.make_d_layer(1024, 512, block_num[4], s=2)
        self.d_layer4 = self.make_d_layer(512, 256, block_num[3], s=2)
        self.d_layer3 = self.make_d_layer(256, 128, block_num[2], s=2)
        self.d_layer2 = self.make_d_layer(128, 64, block_num[1], s=2)
        self.d_layer1 = self.make_d_layer(64, 1, block_num[0], s=2)

        self.f1 = f(4096, 4096)
        self.f2 = f(4096, 4096)

        self.r = r()
        self.t = t()
        self.d = d1()


    def make_c_layer(self, i, o, num_blocks, s, k=3, p=1):
        layers = [c_block(i, o, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(c_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, i, o, num_blocks, s, k=3, p=1):
        layers = [d_block(i, o, k=k, s=s, p=p)]
        for _ in range(num_blocks - 1):
            layers.append(d_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""

        """ encoder """
        x1 = self.c_layer1(x)  # [1,64,64]    >>  [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>  [128,16,16]
        x3 = self.c_layer3(x2)  # [128,16,16]  >>  [256,8,8]
        x4 = self.c_layer4(x3)  # [256,8,8]  >>  [512,4,4]
        x5 = self.c_layer5(x4)  # [512,4,4]  >>  [1024,2,2]

        """ fc layer """
        y1 = x5
        y2 = v1(y1)
        y3 = self.d(self.r(self.f1(y2)))
        y4 = v2(y3, [1024, 2, 2])

        """decoder1"""
        z6 = y4
        z4 = self.d_layer5(z6) + x4  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4) + x3 # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3) + x2 # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2)  # [128,16,16]    >>  [64,32,32]
        z0 = self.d_layer1(z1)  # [64,32,32]  >>  [1,64,64]

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.t(self.f2(w2))
        w4 = v2(w3, [1, 64, 64])

        p1 = w4
        p2 = torch.stack((torch.cos(p1 * math.pi), torch.sin(p1 * math.pi)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        # p5 = n2(p4)

        return p1, p4


class HGNMLP(nn.Module):
    def __init__(self, block_num):
        super(HGNMLP, self).__init__()

        """ Layers """

        self.c_layer1 = self.make_c_layer(64, block_num[0], s=2)
        self.c_layer2 = self.make_c_layer(128, block_num[1], s=2)
        self.c_layer3 = self.make_c_layer(256, block_num[2], s=2)
        self.c_layer4 = self.make_c_layer(512, block_num[3], s=2)
        self.c_layer5 = self.make_c_layer(1024, block_num[4], s=2)
        self.d_layer5 = self.make_d_layer(512, block_num[4], s=2)
        self.d_layer4 = self.make_d_layer(256, block_num[3], s=2)
        self.d_layer3 = self.make_d_layer(128, block_num[2], s=2)
        self.d_layer2 = self.make_d_layer(64, block_num[1], s=2)
        self.d_layer1 = self.make_d_layer(1, block_num[0], s=2)

        self.f1 = f(4096, 4096)
        self.f2 = f(4096, 4096)
        self.f3 = f(4096, 4096)
        self.r = r()
        self.t = t()

    def make_c_layer(self, o, num_blocks, s, k=3, p=1):
        if o == 64:
            layers = [c_block(1, o, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(c_block(o, o, k=k, s=1, p=p))

        else:
            layers = [c_block(o // 2, o, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(c_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, o, num_blocks, s, k=3, p=1):
        if o == 1:
            layers = [d_block(64, 1, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(d_block(1, 1, k=k, s=1, p=p))
        else:
            layers = [d_block(o * 2, o, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(d_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""

        """ encoder """
        x1 = self.c_layer1(x)  # [1,64,64]    >>  [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>  [128,16,16]
        x3 = self.c_layer3(x2)  # [128,16,16]  >>  [256,8,8]
        x4 = self.c_layer4(x3)  # [256,8,8]  >>  [512,4,4]
        x5 = self.c_layer5(x4)  # [512,4,4]  >>  [1024,2,2]

        """ fc layer """
        y1 = x5
        y2 = v1(y1)
        y3 = self.r(self.f1(y2))
        y4 = v2(y3, [1024, 2, 2])

        """decoder1"""
        z6 = y4
        z4 = self.d_layer5(z6)  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4)  # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3)  # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2)  # [128,8,8]    >>  [64,16,16]
        z0 = self.d_layer1(z1)  # [64,16,16]  >>  [1,32,32]

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.r(self.f2(w2))
        w4 = self.t(self.f3(w3))
        w5 = v2(w4, [1, 64, 64])

        p1 = w5
        p2 = torch.stack((torch.cos(p1 * math.pi), torch.sin(p1 * math.pi)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        p5 = n2(p4)

        return p1, p5


class HGN_GAN(nn.Module):
    def __init__(self, block_num):
        super(HGN_GAN, self).__init__()

        """ Layers """

        self.c_layer1 = self.make_c_layer(64, block_num[0], s=2)
        self.c_layer2 = self.make_c_layer(128, block_num[1], s=2)
        self.c_layer3 = self.make_c_layer(256, block_num[2], s=2)
        self.c_layer4 = self.make_c_layer(512, block_num[3], s=2)
        self.c_layer5 = self.make_c_layer(1024, block_num[4], s=2)
        self.d_layer5 = self.make_d_layer(512, block_num[4], s=2)
        self.d_layer4 = self.make_d_layer(256, block_num[3], s=2)
        self.d_layer3 = self.make_d_layer(128, block_num[2], s=2)
        self.d_layer2 = self.make_d_layer(64, block_num[1], s=2)
        self.d_layer1 = self.make_d_layer(1, block_num[0], s=2)

        self.f1 = f(4096, 4096)
        self.f2 = f(4096, 4096)
        self.f3 = f(4096, 4096)

        self.r = r()

    def make_c_layer(self, o, num_blocks, s, k=3, p=1):
        if o == 64:
            layers = [c_block(1, o, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(c_block(o, o, k=k, s=1, p=p))

        else:
            layers = [c_block(o // 2, o, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(c_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, o, num_blocks, s, k=3, p=1):
        if o == 1:
            layers = [d_block(64, 1, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(d_block(1, 1, k=k, s=1, p=p))
        else:
            layers = [d_block(o * 2, o, k=k, s=s, p=p)]
            for _ in range(num_blocks - 1):
                layers.append(d_block(o, o, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""

        """ encoder """
        x1 = self.c_layer1(x)  # [1,64,64]    >>  [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>  [128,16,16]
        x3 = self.c_layer3(x2)  # [128,16,16]  >>  [256,8,8]
        x4 = self.c_layer4(x3)  # [256,8,8]  >>  [512,4,4]
        x5 = self.c_layer5(x4)  # [512,4,4]  >>  [1024,2,2]

        """ fc layer """
        y1 = x5
        y2 = v1(y1)
        y3 = self.r(self.f1(y2))
        y4 = v2(y3, [1024, 2, 2])

        """decoder"""
        z6 = y4
        z4 = self.d_layer5(z6) + x4  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4) + x3  # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3)  # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2)  # [128,8,8]    >>  [64,16,16]
        z0 = self.d_layer1(z1)  # [64,16,16]  >>  [1,32,32]

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.r(self.f2(w2))
        w4 = v2(w3, [1, 64, 64])
        w5 = n(w4)

        """processing"""
        p1 = w5
        p2 = torch.stack((torch.cos(p1 * 2 * math.pi), torch.sin(p1 * 2 * math.pi)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        p5 = n(p4)

        return w5, p5