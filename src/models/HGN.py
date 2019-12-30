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
    def __init__(self, args):
        super(HGN_size, self).__init__()

        blocks = args.block_nums
        self.layer_num = len(blocks)
        self.ch = args.channel_size
        self.sz = args.img_size

        self.c11 = c(1, self.ch, k=3, s=1, p=1)
        self.c12 = c(self.ch, self.ch, k=3, s=1, p=1)
        self.b11 = b(self.ch)
        self.b12 = b(self.ch)

        self.c_layer11 = self.make_c_layer(blocks[0])
        self.c_layer12 = self.make_c_layer(blocks[1])
        self.c_layer13 = self.make_c_layer(blocks[2])
        self.c_layer14 = self.make_c_layer(blocks[3])
        self.c_layer15 = self.make_c_layer(blocks[4])

        self.ch = args.channel_size * 2

        self.c21 = c(1, self.ch, k=3, s=1, p=1)
        self.c22 = c(self.ch, self.ch, k=3, s=1, p=1)
        self.b21 = b(self.ch)
        self.b22 = b(self.ch)

        self.c_layer22 = self.make_c_layer(blocks[1])
        self.c_layer23 = self.make_c_layer(blocks[2])
        self.c_layer24 = self.make_c_layer(blocks[3])
        self.c_layer25 = self.make_c_layer(blocks[4])

        self.ch = args.channel_size * 4

        self.c31 = c(1, self.ch, k=3, s=1, p=1)
        self.c32 = c(self.ch, self.ch, k=3, s=1, p=1)
        self.b31 = b(self.ch)
        self.b32 = b(self.ch)

        self.c_layer33 = self.make_c_layer(blocks[2])
        self.c_layer34 = self.make_c_layer(blocks[3])
        self.c_layer35 = self.make_c_layer(blocks[4])

        self.d_layer5 = self.make_d_layer(blocks[4])
        self.d_layer4 = self.make_d_layer(blocks[3])
        self.d_layer3 = self.make_d_layer(blocks[2])
        self.d_layer2 = self.make_d_layer(blocks[1])
        self.d_layer1 = self.make_d_layer(blocks[0])

        self.c1 = c(self.ch, self.ch, k=3, s=1, p=1)
        self.c2 = c(self.ch, 1, k=3, s=1, p=1)
        self.b1 = b(self.ch)

        self.var = pow(2, self.layer_num)
        self.ch_num = self.ch * self.var
        self.filter_sz = self.sz // self.var
        self.h_node = self.ch_num * self.filter_sz * self.filter_sz

        self.f1 = f(self.h_node, self.h_node)
        self.f2 = f(self.h_node, self.h_node)
        self.f3 = f(self.sz * self.sz, self.sz * self.sz)

        self.r = r()
        self.t = t()
        self.do = do1()
        self.ds = ds()

    def make_c_layer(self, num_blocks, k=3, s=2, p=1):
        layers = [c_block(self.ch, self.ch * 2, k=k, s=s, p=p)]
        self.ch = self.ch * 2
        for _ in range(num_blocks - 1):
            layers.append(c_block(self.ch, self.ch, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, num_blocks, k=3, s=2, p=1):
        layers = [d_block(self.ch, self.ch // 2, k=k, s=s, p=p)]
        self.ch = self.ch // 2
        for _ in range(num_blocks - 1):
            layers.append(d_block(self.ch, self.ch, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""

        x10 = x
        x10 = self.r(self.b11(self.c11(x10)))
        x10 = self.r(self.b12(self.c12(x10)))

        x20 = self.ds(x)
        x20 = self.r(self.b21(self.c21(x20)))
        x20 = self.r(self.b22(self.c22(x20)))

        x30 = self.ds(self.ds(x))
        x30 = self.r(self.b31(self.c31(x30)))
        x30 = self.r(self.b32(self.c32(x30)))

        """ encoder """
        x11 = self.c_layer11(x10)  # [1,64,64]    >>  [64,32,32]
        x12 = self.c_layer12(x11)  # [64,32,32]   >>  [128,16,16]
        x13 = self.c_layer13(x12)  # [128,16,16]  >>  [256,8,8]
        x14 = self.c_layer14(x13)  # [256,8,8]  >>  [512,4,4]
        x15 = self.c_layer15(x14)  # [512,4,4]  >>  [1024,2,2]

        x21 = self.c_layer22(x20)  # [64,32,32]   >>  [128,16,16]
        x22 = self.c_layer23(x21)  # [128,16,16]  >>  [256,8,8]
        x23 = self.c_layer24(x22)  # [256,8,8]  >>  [512,4,4]
        x24 = self.c_layer25(x23)  # [512,4,4]  >>  [1024,2,2]

        x31 = self.c_layer33(x30)  # [128,16,16]  >>  [256,8,8]
        x32 = self.c_layer34(x31)  # [256,8,8]  >>  [512,4,4]
        x33 = self.c_layer35(x32)  # [512,4,4]  >>  [1024,2,2]

        """ fc layer """
        # y1 = torch.cat((x15, x24, x33), dim=1)
        y1 = x15 + x24 + x33
        y2 = v1(y1)
        y3 = self.do(self.r(self.f1(y2)))
        y4 = self.do(self.r(self.f2(y3)))
        y5 = v2(y4, [self.ch_num, self.filter_sz, self.filter_sz])

        """decoder1"""
        z5 = y5
        z4 = self.d_layer5(z5 + x15 + x24 + x33)  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4 + x14 + x23 + x32)  # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3 + x13 + x22 + x31)  # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2 + x12 + x21)  # [128,16,16]    >>  [64,32,32]
        z0 = self.d_layer1(z1 + x11)  # [64,32,32]  >>  [1,64,64]

        z0 = self.r(self.b1(self.c1(z0)))
        z0 = self.t(self.c2(z0))

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.t(self.f3(w2))
        w4 = v2(w3, [1, self.sz, self.sz])

        p1 = w4
        p2 = torch.stack((torch.cos(p1 * math.pi), torch.sin(p1 * math.pi)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))

        return p1, p4


class HGN(nn.Module):
    def __init__(self, args):
        super(HGN, self).__init__()

        blocks = args.block_nums
        self.layer_num = len(blocks)
        self.ch = args.channel_size
        self.sz = args.img_size

        self.c1 = c(1, self.ch, k=3, s=1, p=1)
        self.c2 = c(self.ch, self.ch, k=3, s=1, p=1)
        self.b1 = b(self.ch)
        self.b2 = b(self.ch)

        self.c_layer1 = self.make_c_layer(blocks[0])
        self.c_layer2 = self.make_c_layer(blocks[1])
        self.c_layer3 = self.make_c_layer(blocks[2])
        self.c_layer4 = self.make_c_layer(blocks[3])
        self.c_layer5 = self.make_c_layer(blocks[4])

        self.d_layer5 = self.make_d_layer(blocks[4])
        self.d_layer4 = self.make_d_layer(blocks[3])
        self.d_layer3 = self.make_d_layer(blocks[2])
        self.d_layer2 = self.make_d_layer(blocks[1])
        self.d_layer1 = self.make_d_layer(blocks[0])

        self.c3 = c(self.ch, self.ch, k=3, s=1, p=1)
        self.c4 = c(self.ch, 1, k=3, s=1, p=1)
        self.b3 = b(self.ch)

        self.var = pow(2, self.layer_num)
        self.ch_num = self.ch * self.var
        self.filter_sz = self.sz // self.var
        self.h_node = self.ch_num * self.filter_sz * self.filter_sz

        self.f1 = f(self.h_node, self.h_node // 4)
        self.f2 = f(self.h_node // 4, self.h_node)
        self.f3 = f(self.sz * self.sz, self.sz * self.sz)

        self.r = r()
        self.t = t()
        self.do = do1()

    def make_c_layer(self, num_blocks, k=3, s=2, p=1):
        layers = [c_block(self.ch, self.ch * 2, k=k, s=s, p=p)]
        self.ch = self.ch * 2
        for _ in range(num_blocks - 1):
            layers.append(c_block(self.ch, self.ch, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def make_d_layer(self, num_blocks, k=3, s=2, p=1):
        layers = [d_block(self.ch, self.ch // 2, k=k, s=s, p=p)]
        self.ch = self.ch // 2
        for _ in range(num_blocks - 1):
            layers.append(d_block(self.ch, self.ch, k=k, s=1, p=p))

        net = nn.Sequential(*layers)
        net.apply(init_weights)
        return net

    def forward(self, x):
        """ forward function"""

        x0 = x
        x0 = self.r(self.b1(self.c1(x0)))
        x0 = self.r(self.b2(self.c2(x0)))

        """ encoder """
        x1 = self.c_layer1(x0)  # [1,64,64]    >>  [64,32,32]
        x2 = self.c_layer2(x1)  # [64,32,32]   >>  [128,16,16]
        x3 = self.c_layer3(x2)  # [128,16,16]  >>  [256,8,8]
        x4 = self.c_layer4(x3)  # [256,8,8]  >>  [512,4,4]
        x5 = self.c_layer5(x4)  # [512,4,4]  >>  [1024,2,2]

        """ fc layer """
        y1 = x5
        y2 = v1(y1)
        y3 = self.do(self.r(self.f1(y2)))
        y4 = self.do(self.r(self.f2(y3)))
        y5 = v2(y4, [self.ch_num, self.filter_sz, self.filter_sz])

        """decoder1"""
        z5 = y5
        z4 = self.d_layer5(z5 + x5)  # [1024,2,2]    >>  [512,4,4]
        z3 = self.d_layer4(z4 + x4)  # [512,4,4]    >>  [256,8,8]
        z2 = self.d_layer3(z3 + x3)  # [256,8,8]    >>  [128,16,16]
        z1 = self.d_layer2(z2 + x2)  # [128,16,16]    >>  [64,32,32]
        z0 = self.d_layer1(z1 + x1)  # [64,32,32]  >>  [1,64,64]

        z0 = self.r(self.b3(self.c3(z0)))
        z0 = self.t(self.c4(z0))

        """ fc layer """
        w1 = z0
        w2 = v1(w1)
        w3 = self.t(self.f3(w2))
        w4 = v2(w3, [1, self.sz, self.sz])

        p1 = w4
        p2 = torch.stack((torch.cos(p1 * math.pi), torch.sin(p1 * math.pi)), dim=4)
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        # p5 = n2(p4)

        return p1, p4


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
        p2 = torch.stack(
            (torch.cos(p1 * 2 * math.pi), torch.sin(p1 * 2 * math.pi)), dim=4
        )
        p3 = torch.fft(p2, 2)
        p4 = torch.sqrt(p3[:, :, :, :, 0].pow(2) + p3[:, :, :, :, 1].pow(2))
        p5 = n2(p4)

        return w5, p5
