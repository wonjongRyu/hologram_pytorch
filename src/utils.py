import csv, time
from os.path import join
from glob import glob
from ops import LayerActivations
import numpy as np
import torch, cv2, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualize_conv_layer(epoch, model):
    img = imread("C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/kaist.png")
    t = torch.from_numpy(np.reshape(img, (1, 1, 64, 64)))

    conv_out = LayerActivations(model.c_layer1, 0)
    o = model(t.cuda())
    conv_out.remove()
    act = conv_out.features
    act = act.cpu().detach().numpy()
    for i in range(32):
        save_path = "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/layers/layer1_"+str(i+1)+'_'+str(epoch)+'.png'
        imwrite(act[0][i], save_path)

    conv_out = LayerActivations(model.c_layer2, 0)
    o = model(t.cuda())
    conv_out.remove()
    act = conv_out.features
    act = act.cpu().detach().numpy()
    for i in range(16):
        save_path = "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/layers/layer2_"+str(i+1)+'_'+str(epoch)+'.png'
        imwrite(act[0][i], save_path)

    conv_out = LayerActivations(model.c_layer3, 0)
    o = model(t.cuda())
    conv_out.remove()
    act = conv_out.features
    act = act.cpu().detach().numpy()
    for i in range(8):
        save_path = "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/layers/layer3_"+str(i+1)+'_'+str(epoch)+'.png'
        imwrite(act[0][i], save_path)

"""check arguments"""


def check_args(args):
    # --checkpoint_dir
    make_result_folder(args)
    make_loss_file(args)
    return args


"""gs algorithm"""


def add_random_phase(img):
    size = img.shape[0]
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(size, size))  # random phase
    img_with_random_phase = np.multiply(img, random_phase)
    return img_with_random_phase


def gs_algorithm(img, iteration_num):
    """rgb2gray"""

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
        holo1, holo100 = get_gs_10and100(img)
        imwrite(holo1, os.path.join(train_image, "gs1/"+str(i+1)+".png"))
        imwrite(holo100, os.path.join(train_image, "gs100/"+str(i+1)+".png"))

    for i in range(len_valid):
        img = imread(os.path.join(valid_image, "images/" + str(i + 1) + ".png"))
        holo1, holo100 = get_gs_10and100(img)
        imwrite(holo1, os.path.join(valid_image, "gs1/"+str(i+1)+".png"))
        imwrite(holo100, os.path.join(valid_image, "gs100/" + str(i + 1) + ".png"))

    for i in range(len_test):
        img = imread(os.path.join(test_image, "images/" + str(i + 1) + ".png"))
        holo1, holo100 = get_gs_10and100(img)
        imwrite(holo1, os.path.join(test_image, "gs1/"+str(i+1)+".png"))
        imwrite(holo100, os.path.join(test_image, "gs100/" + str(i + 1) + ".png"))


""" print """


"""
def print_loss(epoch, seconds, train_loss, valid_loss):
    h = int(seconds / 3600)
    seconds = seconds - h * 3600
    m = int(seconds / 60)
    seconds = seconds - m * 60
    s = int(seconds)
    print(
        f"[epoch:{epoch:04}] time:{h:02}h {m:02}m {s:02}s, train_loss:{train_loss:.04}, valid_loss:{valid_loss:.04}"
    )
"""


def print_loss(epoch, seconds, train_loss, valid_loss):
    h, m, s = get_hms(seconds)
    if epoch == 1:
        print("epoch, time, train_loss_holo, valid_loss_holo, train_loss_image, valid_loss_image")
    print(f"[{epoch:04}] {h:02}h{m:02}m{s:02}s, {train_loss[0]:.04}, {valid_loss[0]:.04}, {train_loss[1]:.04}, {valid_loss[1]:.04}")


def print_start_time():
    tl, ml = get_time_list()
    print('')
    print('=====================[   {}{} {}:{}   ]====================='.format(ml[int(tl[0])-1], tl[1], tl[2], tl[3]))
    print('=====================[   TRAIN START   ]=====================')
    print('')


""" time """


def get_time_list():
    now = time.localtime(time.time())
    time_list = [now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min]
    time_list = list(map(str, time_list))
    for i in range(len(time_list)):
        if len(time_list[i]) != 2:
            time_list[i] = '0' + time_list[i]
    month_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return time_list, month_list


def get_hms(seconds):
    h = int(seconds / 3600)
    seconds = seconds - h * 3600
    m = int(seconds / 60)
    seconds = seconds - m * 60
    s = int(seconds)
    return h, m, s


""" directories """


def make_result_folder(args):
    tl, _ = get_time_list()
    args.save_image_path = '../results/' + tl[0] + '_' + tl[1] + '_' + tl[2] + '_' + tl[3]
    check_folder(args.save_image_path)


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


""" ckpt """


def ckpt(model):
    # Load checkpoint.
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    model.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    epoch_ckpt = checkpoint["epoch"]
    return model, best_acc, epoch_ckpt


""" csv record """


def make_loss_file(args):
    tl, _ = get_time_list()
    args.save_loss_path = '../results/' + tl[0] + '_' + tl[1] + '_' + tl[2] + '_' + tl[3] + '.csv'
    f = open(args.save_loss_path, "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow(["epoch", "time", "train_loss_holo", "train_loss_image", "valid_loss_holo", "valid_loss_image"])
    f.close()


def record_loss(args, epoch, seconds, train_loss, valid_loss):
    h, m, s = get_hms(seconds)
    hms = str(h) + 'h' + str(m) + 'm' + str(s) + 's'
    f = open(args.save_loss_path, "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow([epoch, hms, train_loss[0], valid_loss[0], train_loss[1], valid_loss[1]])
    f.close()


""" image """


def normalize_img(img):
    if (np.max(img) - np.min(img)) == 0:
        return img
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def imread(img_path):
    img = mpimg.imread(img_path)
    return img.astype("float32")


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
