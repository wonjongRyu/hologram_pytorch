from utils import *
import time
import numpy as np

# make_phase_projection("../../dataset4000")

# make_phase_projection("../../test_gs_1to100")

# make_holograms("../../dataset/4000")

since = time.time()

img = imread("../kaist1000.png")

"""Add Random Phase"""
hologram = np.fft.ifft2(img)

"""Iteration"""
for i in range(100):
    reconimg = np.fft.fft2(np.exp(1j * np.angle(hologram)))
    hologram = np.fft.ifft2(np.multiply(img, np.exp(1j * np.angle(reconimg))))

"""Normalization"""
hologram = normalize_img(np.angle(hologram))

print(time.time()-since)
