import csv, time, math
from glob import glob
from ops import LayerActivations
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
from collections import OrderedDict


""" Make output directories """


def get_size(img):
    return np.shape(img)[0]


def get_filtered_img(img, model, layer):
    conv_out = LayerActivations(layer, 1)
    o = model(img)
    conv_out.remove()
    act = conv_out.features
    act = act.cpu().detach().numpy()
    filtered_img = act[0]
    return filtered_img


def combine(*args):
    combined_str = []
    for arg in args:
        combined_str.append(str(arg))
    combined_str = "".join(combined_str)
    return combined_str


def save_start_time(args):
    tl = get_time_list()
    args.since = tl[0] + tl[1] + tl[2] + "_" + tl[3] + tl[4]
    print("")
    print("=" * 25 + "[   " + args.since + "   ]" + "=" * 25)
    print("=" * 25 + "[   TRAIN_START   ]" + "=" * 25)
    print("")


def write_csv_row(csv_path, word_list):
    f = open(csv_path, "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow(word_list)
    f.close()


def make_csvfile_and_folders(args):
    args.save_path_of_outputs = os.path.join(args.save_path_of_outputs, args.since)
    check_and_make_folder(args.save_path_of_outputs)
    make_csvfile(args)
    make_folders(args)


def make_folders(args):
    make_images_folder(args)
    make_models_folder(args)
    make_layers_folder(args)
    make_tensor_folder(args)


def make_csvfile(args):
    make_loss_file(args)
    make_arch_file(args)
    make_args_file(args)


def make_images_folder(args):
    args.save_path_of_images = args.save_path_of_outputs + "/images"
    check_and_make_folder(args.save_path_of_images)


def make_models_folder(args):
    args.save_path_of_models = args.save_path_of_outputs + "/models"
    check_and_make_folder(args.save_path_of_models)


def make_layers_folder(args):
    args.save_path_of_layers = args.save_path_of_outputs + "/layers"
    check_and_make_folder(args.save_path_of_layers)


def make_tensor_folder(args):
    args.save_path_of_tensor = args.save_path_of_outputs + "/tensor"
    check_and_make_folder(args.save_path_of_tensor)


def make_loss_file(args):
    args.save_path_of_loss = args.save_path_of_outputs + "/loss_" + args.since + ".csv"
    write_csv_row(args.save_path_of_loss, ["epoch", "time", "train_loss", "valid_loss"])


def make_arch_file(args):
    args.save_path_of_arch = args.save_path_of_outputs + "/arch_" + args.since + ".csv"
    write_csv_row(args.save_path_of_arch, [""])


def make_args_file(args):
    args.save_path_of_args = args.save_path_of_outputs + "/args_" + args.since + ".csv"

    args_dict = vars(args)
    args_keys = list(args_dict.keys())
    args_vals = list(args_dict.values())
    for i in range(len(args_keys)):
        write_csv_row(args.save_path_of_args, [args_keys[i], args_vals[i]])


def check_and_make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


""" etc """


def make_labels(batch_size):
    real_label = torch.FloatTensor(batch_size)
    real_label = fill_label(real_label, 1)
    real_label = real_label.cuda()

    fake_label = torch.FloatTensor(batch_size)
    fake_label = fill_label(fake_label, 0)
    fake_label = fake_label.cuda()
    return real_label, fake_label


def fill_label(label, value):
    for i in range(len(label)):
        label[i] = value
    return label


def save_tensor(args, tensors):
    save_root = args.save_path_of_tensor
    for i in range(len(tensors)):
        file_path = combine('tensor', i+1, '.csv')
        save_path = os.path.join(save_root, file_path)
        np.savetxt(save_path, tensors[i, :, :], delimiter=",")


def visualize_conv_layer(args, model):
    img_path = "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/"
    img_list = ["lenna.png", "flower.png", "kaist.png"]
    save_root = args.save_path_of_layers

    for k in range(3):

        img = imread(img_path + img_list[k])
        sz = get_size(img)
        img = torch.from_numpy(np.reshape(img, (1, 1, sz, sz)))
        img = img.cuda()

        c_layer = [
            model.c_layer1,
            model.c_layer2,
            model.c_layer3,
            model.c_layer4,
            model.c_layer5,
        ]
        d_layer = [
            model.d_layer1,
            model.d_layer2,
            model.d_layer3,
            model.d_layer4,
            model.d_layer5,
        ]

        for i in range(5):
            layer = c_layer[i]
            images = get_filtered_img(img, model, layer)
            for j in range(len(images)):
                file_path = combine("image", k + 1, "_layer", i + 1, "_", j + 1, ".png")
                save_path = os.path.join(save_root, file_path)
                imwrite(images[j], save_path)

        for i in range(5):
            layer = d_layer[4-i]
            images = get_filtered_img(img, model, layer)
            for j in range(len(images)):
                file_path = combine("image", k + 1, "_layer", i + 6, "_", j + 1, ".png")
                save_path = os.path.join(save_root, file_path)
                imwrite(images[j], save_path)


def visualize_conv_filters(model):
    model.state_dict().keys()
    cnn_weights = model.state_dict()["c_layer1.0.c1.weight"].cpu()
    cnn_weights = cnn_weights.cpu().detach().numpy()
    for i in range(64):
        save_path = (
            "C:/Users/CodeLab/PycharmProjects/hologram_pytorch/hologram_pytorch/filters/layer1_"
            + str(i + 1)
            + ".png"
        )
        imwrite(cnn_weights[i][0], save_path)



""" print """


def print_loss(epoch, seconds, train_loss, valid_loss):
    h, m, s = get_hms(seconds)
    if epoch == 1:
        print("[epoch] [  time  ] [train] [valid]")
    print(f"[{epoch:05}] {h:02}h{m:02}m{s:02}s, {train_loss:.03}, {valid_loss:.03}")


""" time """


def get_time_list():
    now = time.localtime(time.time())
    time_list = [now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min]
    time_list[0] = time_list[0] - 2000
    time_list = list(map(str, time_list))
    for i in range(len(time_list)):
        if len(time_list[i]) == 1:
            time_list[i] = "0" + time_list[i]
    return time_list


def get_hms(seconds):
    h = int(seconds / 3600)
    seconds = seconds - h * 3600
    m = int(seconds / 60)
    seconds = seconds - m * 60
    s = int(seconds)
    return h, m, s


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


def record_on_csv(args, epoch, seconds, train_loss, valid_loss):
    h, m, s = get_hms(seconds)
    hms = str(h) + "h" + str(m) + "m" + str(s) + "s"
    f = open(args.save_path_of_loss, "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow([epoch, hms, train_loss, valid_loss])
    f.close()


""" image """


def imnorm(img):
    if (np.max(img) - np.min(img)) == 0:
        return img
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def imread(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255
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
    img = img * 255
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


def imresize(img, w, h):
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def imrotate90(img):
    img = cv2.transpose(img)
    img = cv2.flip(img, 1)
    return img


def summary(args, model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    f = open(args.save_path_of_arch, "a", encoding="utf-8", newline="")
    wr = csv.writer(f)
    wr.writerow(["Layer (type)", "Output Shape", "Param #"])
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = [
            layer,
            summary[layer]["output_shape"],
            int(summary[layer]["nb_params"]),
        ]
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        wr.writerow(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4.0 / (1024 ** 2.0))
    total_output_size = abs(
        2.0 * total_output * 4.0 / (1024 ** 2.0)
    )  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4.0 / (1024 ** 2.0))
    total_size = total_params_size + total_output_size + total_input_size

    wr.writerow(["================================================================"])
    wr.writerow(["Total params", total_params.item()])
    wr.writerow(["Total params", total_params.item()])
    wr.writerow(["Trainable params", trainable_params.item()])
    wr.writerow(["Non-trainable params", (total_params - trainable_params).item()])
    wr.writerow(["================================================================"])
    wr.writerow(["Input size (MB)", total_input_size.item()])
    wr.writerow(["Forward/backward pass size (MB)", total_output_size.item()])
    wr.writerow(["Params size (MB)", total_params_size.item()])
    wr.writerow(["Estimated Total Size (MB)", total_size.item()])
    wr.writerow(["================================================================"])

    f.close()
