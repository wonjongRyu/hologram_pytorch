import cv2, torch
from utils import *

img_path = "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/"
img_list = ["lenna.png", "flower.png", "kaist.png"]

images = np.ndarray([3, 64, 64], dtype=float)
ze = torch.zeros([3, 64, 64], dtype=float)
on = torch.ones([3, 64, 64], dtype=float)
for i in range(3):
    img = imread(img_path + img_list[i])
    images[i, :, :] = img

images = torch.from_numpy(images)
images1 = torch.stack((images, ze), dim=3)
images1 = torch.ifft(images1, 2)
images1 = torch.sqrt(images1[:, :, :, 0].pow(2) + images1[:, :, :, 1].pow(2))
images1 = torch.sum(images1, dim=2)
images1 = torch.sum(images1, dim=1)

images1 = torch.reshape(images1, [3,1,1])
print(images1)
print(images)
print(torch.div(images, images1)*4096)