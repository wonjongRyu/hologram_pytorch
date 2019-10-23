import os, csv
import numpy as np
import torch, cv2, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


"""check arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.save_image_path)

    # --result_dir
    check_folder(args.save_model_path)
    return args


"""check directory"""


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def ckpt(model):
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    epoch_ckpt = checkpoint["epoch"]
    return model, best_acc, epoch_ckpt


def visualize_graph(train_loss, valid_loss):
    plt.plot(range(1, len(train_loss) + 1), train_loss, "bo", label="train error")
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, "r", label="valid error")
    plt.legend()
    plt.show()


def print_loss(epoch, time, train_epoch_loss, valid_epoch_loss):
    print(
        f"[epoch:{epoch}] time:{time:{4}.{4}}, train loss:{train_epoch_loss:{5}.{4}}, valid loss:{valid_epoch_loss:{5}.{4}}"
    )


def get_psnr(a, b):
    psnr = 0
    error = a - b
    for i in range(error.shape[0]):
        img = error[i, :, :, 0]
        mse = np.mean(np.square(img))
        if mse == 0:
            psnr = psnr + 100
        else:
            psnr = psnr - 20 * np.log10(np.sqrt(mse))
    psnr = psnr / error.shape[0]
    return psnr


def csv_initial():
    f = open("../output.csv", "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow(["epoch", "time", "loss_train", "loss_test", "psnr_train", "psnr_test"])
    f.close()


def csv_record(epoch, time, rsme_train, rsme_test, psnr_train, psnr_test):
    f = open("../output.csv", "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow([epoch, time, rsme_train, psnr_train, rsme_test, psnr_test])
    f.close()


def csv_record_hologram(epoch, hologram):
    f = open("../hologram.csv", "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow([epoch])
    for i in range(8):
        for j in range(8):
            wr.writerow([hologram[i][j]])
    f.close()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def normalize_img(img):
    if (np.max(img) - np.min(img)) == 0:
        return img/img
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def imread(img_path, index=1):
    img = mpimg.imread(img_path)
    if (np.max(img) - np.min(img)) == 0:
        print(index)
    img = img

    return img.astype("float32")


def imresize(img, size):
    height = img.shape[0]
    if height > size[0]:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    return img


def imshow(img):
    if len(img.shape) == 2:
        """Gray Image"""
        plt.imshow(img, cmap="gray")
        plt.show()
    else:
        """RGB Image"""
        plt.imshow(img)
        plt.show()


def imwrite(img, save_path):
    img = normalize_img(img) * 255
    cv2.imwrite(save_path, img)


def histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 1])
    plt.hist(img.ravel(), 256, [0, 1])
    plt.show()


def add_random_phase(img):
    size = img.shape[0]
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(size, size))  # random phase
    img_with_random_phase = np.multiply(img, random_phase)
    return img_with_random_phase


def gs_algorithm(img, iteration_num):
    """rgb2gray"""
    if len(img.shape) == 3:
        img = rgb2gray(img)

    """Add Random Phase"""
    img_with_random_phase = add_random_phase(img)
    hologram = np.fft.ifft2(img_with_random_phase)

    """Iteration"""
    for i in range(iteration_num):
        reconimg = np.fft.fft2(np.exp(1j * np.angle(hologram)))
        hologram = np.fft.ifft2(np.multiply(img, np.exp(1j * np.angle(reconimg))))

    """Normalization"""
    hologram = normalize_img(np.angle(hologram))
    reconimg = normalize_img(np.abs(reconimg))
    return hologram, reconimg


def get_psnr(a, b):
    psnr = 0
    errors = a - b
    for i in range(np.shape(a)[0]):
        error = errors[i, :, :, 0]
        mse = np.mean(error ** 2)
        if mse == 0:
            psnr = psnr + 100
        else:
            psnr = psnr - 20 * np.log10(np.sqrt(mse))
    return psnr / np.shape(a)[0]


def high_pass_filter(img):
    filter = np.array(
        [
            [0, -1, -1, -1, 0],
            [-1, 2, -4, 2, -1],
            [-1, -4, 21, -4, -1],
            [-1, 2, -4, 2, -1],
            [0, -1, -1, -1, 0],
        ],
        dtype="f",
    )
    filtered_img = signal.convolve2d(img, filter, "same")
    filtered_img = (filtered_img - np.min(filtered_img)) / (
        np.max(filtered_img) - np.min(filtered_img)
    )
    return filtered_img
