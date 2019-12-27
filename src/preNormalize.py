import numpy
from utils import *

sz = 64
img_path = "../kaist.png"
img = imread(img_path)
img = imresize(img, sz, sz)
hologram, reconimg = gs_algorithm(img, 1000)

reconimg = np.fft.fft2(hologram)
hologram1 = np.sum(abs(hologram))
reconimg1 = np.sum(abs(reconimg))
max1 = np.max(abs(reconimg))

hologram = np.exp(1j*np.angle(hologram))
reconimg = np.fft.fft2(hologram)

hologram2 = np.sum(abs(hologram))
reconimg2 = np.sum(abs(reconimg))
max2 = np.max(abs(reconimg))

print(hologram1)
print(hologram2)
print(reconimg1)
print(reconimg2)
print(max1)
print(max2)
print(hologram2/hologram1)
print(reconimg2/reconimg1)
print(max2/max1)
print(hologram1.dtype)


