from torchvision.models import resnet18, resnet34
import torch.nn as nn
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)






def get_resnet_features(is_cuda, train_loader, valid_loader, block_num=18):
    if block_num == 18:
        pre_resnet = resnet18(pretrained=True)

    if block_num == 34:
        pre_resnet = resnet34(pretrained=True)

    if is_cuda:
        pre_resnet = pre_resnet.cuda()

    for p in pre_resnet.parameters():
        p.requires_grad = False

    model = nn.Sequential(*list(pre_resnet.children())[:-1])

    train_labels = []
    train_features = []

    for data, label in train_loader:
        o = model(Variable(data.cuda()))
        o = o.view(o.size(0), -1)
        train_labels.extend(label)
        train_features.extend(o.cpu().data)

    valid_labels = []
    valid_features = []

    for data, label in valid_loader:
        o = model(Variable(data.cuda()))
        o = o.view(o.size(0), -1)
        valid_labels.extend(label)
        valid_features.extend(o.cpu().data)

    pretrained_resnet_features = [
        train_features,
        train_labels,
        valid_features,
        valid_labels
    ]

    return pretrained_resnet_features
