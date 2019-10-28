import os, csv, time
from glob import glob
import numpy as np
import torch, cv2, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


"""check arguments"""


def check_args(args):
    # --checkpoint_dir
    make_result_folder(args)

    # --result_dir
    check_folder(args.save_model_path)
    return args


"""check directory"""


def make_holograms(dataset_path):
    train_image = os.path.join(dataset_path, "train")
    valid_image = os.path.join(dataset_path, "valid")
    test_image = os.path.join(dataset_path, "test")

    len_train = len(glob(os.path.join(train_image, "images/*.*")))
    len_valid = len(glob(os.path.join(valid_image, "images/*.*")))
    len_test = len(glob(os.path.join(test_image, "images/*.*")))

    for i in range(len_train):
        img = imread(os.path.join(train_image, "images/" + str(i+1)+".png"))
        hologram = gs_algorithm(img, 10)
        imwrite(hologram, os.path.join(train_image, "holograms/"+str(i+1)+".png"))

    for i in range(len_valid):
        img = imread(os.path.join(valid_image, "images/" + str(i + 1) + ".png"))
        hologram = gs_algorithm(img, 10)
        imwrite(hologram, os.path.join(valid_image, "holograms/" + str(i + 1) + ".png"))

    for i in range(len_test):
        img = imread(os.path.join(test_image, "images/" + str(i + 1) + ".png"))
        hologram = gs_algorithm(img, 10)
        imwrite(hologram, os.path.join(test_image, "holograms/" + str(i + 1) + ".png"))


def make_fft_phase(dataset_path):
    train_image = os.path.join(dataset_path, "train")
    valid_image = os.path.join(dataset_path, "valid")
    test_image = os.path.join(dataset_path, "test")

    len_train = len(glob(os.path.join(train_image, "images/*.*")))
    len_valid = len(glob(os.path.join(valid_image, "images/*.*")))
    len_test = len(glob(os.path.join(test_image, "images/*.*")))

    for i in range(len_train):
        img = imread(os.path.join(train_image, "images/" + str(i+1)+".png"))
        hologram = gs_algorithm(img, 100)
        imwrite(hologram, os.path.join(train_image, "holograms/"+str(i+1)+".png"))

    for i in range(len_valid):
        img = imread(os.path.join(valid_image, "images/" + str(i + 1) + ".png"))
        hologram = gs_algorithm(img, 100)
        imwrite(hologram, os.path.join(valid_image, "holograms/" + str(i + 1) + ".png"))

    for i in range(len_test):
        img = imread(os.path.join(test_image, "images/" + str(i + 1) + ".png"))
        hologram = gs_algorithm(img, 100)
        imwrite(hologram, os.path.join(test_image, "holograms/" + str(i + 1) + ".png"))


def make_phase_projection(dataset_path):
    train_image = os.path.join(dataset_path, "train")
    valid_image = os.path.join(dataset_path, "valid")
    test_image = os.path.join(dataset_path, "test")

    len_train = len(glob(os.path.join(train_image, "images/*.*")))
    len_valid = len(glob(os.path.join(valid_image, "images/*.*")))
    len_test = len(glob(os.path.join(test_image, "images/*.*")))

    for i in range(len_train):
        img = imread(os.path.join(train_image, "images/" + str(i+1)+".png"))
        holo10, holo100 = get_gs_10and100(img)
        imwrite(holo10, os.path.join(train_image, "images/"+str(i+1)+".png"))
        imwrite(holo100, os.path.join(train_image, "holograms/"+str(i+1)+".png"))

    for i in range(len_valid):
        img = imread(os.path.join(valid_image, "images/" + str(i + 1) + ".png"))
        holo10, holo100 = get_gs_10and100(img)
        imwrite(holo10, os.path.join(train_image, "images/"+str(i+1)+".png"))
        imwrite(holo100, os.path.join(valid_image, "holograms/" + str(i + 1) + ".png"))

    for i in range(len_test):
        img = imread(os.path.join(test_image, "images/" + str(i + 1) + ".png"))
        holo10, holo100 = get_gs_10and100(img)
        imwrite(holo10, os.path.join(train_image, "images/"+str(i+1)+".png"))
        imwrite(holo100, os.path.join(test_image, "holograms/" + str(i + 1) + ".png"))


def get_time_list():
    now = time.localtime(time.time())
    time_list = [now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min]
    time_list = list(map(str, time_list))
    time_list[0] = time_list[0][2:]
    for i in range(1, 5):
        if len(time_list[i]) != 2:
            time_list[i] = '0' + time_list[i]
    time_list = time_list + ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return time_list


def print_start_time():
    tl = get_time_list()
    print('')
    print('=================[   {}{} {}:{}   ]================='.format(tl[4+int(tl[1])], tl[2], tl[3], tl[4]))
    print('=================[   TRAIN START   ]=================')
    print('')


def make_result_folder(args):
    time_list = get_time_list()
    save_dir = '../results/images'
    for i in range(0, 5):
        save_dir = save_dir + '_' + time_list[i]

    args.save_image_path = save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


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


def print_loss(epoch, seconds, train_loss, valid_loss):
    h = int(seconds / 3600)
    seconds = seconds - h * 3600
    m = int(seconds / 60)
    seconds = seconds - m * 60
    s = int(seconds)
    print(
        f"[epoch:{epoch:04}] time:{h:02}h {m:02}m {s:02}s, train_loss:{train_loss:.04}, valid_loss:{valid_loss:.04}"
    )


def print_2_loss(epoch, seconds, train_loss1, train_loss2, train_total, valid_loss1, valid_loss2, valid_total):
    h = int(seconds / 3600)
    seconds = seconds - h * 3600
    m = int(seconds / 60)
    seconds = seconds - m * 60
    s = int(seconds)
    if epoch == 0:
        print("epoch, time, train_loss1, train_loss2, train_total, valid1, valid2, valid_total")

    print(f"[{epoch:04}] {h:02}h {m:02}m {s:02}s, {train_loss1:.04}, {train_loss2:.04}, {train_total:.04}, {valid_loss1:.04}, {valid_loss2:.04}, {valid_total:.04}")


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
    # reconimg = normalize_img(np.abs(reconimg))
    # return hologram, reconimg
    return hologram


def get_gs_10and100(img):
    """rgb2gray"""

    """Add Random Phase"""
    img_with_random_phase = add_random_phase(img)
    hologram = np.fft.ifft2(img_with_random_phase)

    holo1 = normalize_img(np.angle(hologram))

    """Iteration"""

    for i in range(100):
        reconimg = np.fft.fft2(np.exp(1j * np.angle(hologram)))
        hologram = np.fft.ifft2(np.multiply(img, np.exp(1j * np.angle(reconimg))))

    """Normalization"""
    holo100 = normalize_img(np.angle(hologram))

    return holo1, holo100


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
