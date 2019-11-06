import numpy as np
from utils import *


for i in range(1, 3201):
    img = imread("C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/dataset4000/train/holos/" + str(i) + '.png')
    imwrite(img, "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/hc/train/" + str(2*i-1) + '.png')

for i in range(1, 3201):
    img = np.random.random((64, 64))
    imwrite(img, "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/hc/train/" + str(2*i) + '.png')

for i in range(1, 641):
    img = imread("C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/dataset4000/train/holos/" + str(i) + '.png')
    imwrite(img, "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/hc/test/" + str(2*i-1) + '.png')

for i in range(1, 641):
    img = np.random.random((64, 64))
    imwrite(img, "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/hc/test/" + str(2*i) + '.png')

