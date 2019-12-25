import argparse
from models.HGN import HGN_sincos
from data import *
import time


def parse_args():
    """ parser_args function """

    parser = argparse.ArgumentParser(description="Hologram Generation Net")

    """ Dataset Path """
    parser.add_argument("--path_of_dataset", type=str, default="../dataset/4000")

    """ Training Condition """
    parser.add_argument("--list_of_block_numbers", type=list, default=[2, 2, 2, 2, 2])

    return check_args(parser.parse_args())


def main():
    """ Load Arguments """
    args = parse_args()

    """ Define Network """
    G = HGN_sincos(args.list_of_block_numbers)
    G.load_state_dict(torch.load("../models/HGS_sincos_train_finished.pt"))

    # print(G)

    """ test model """
    model_test(G)

    """ visualize images  """
    visualize_conv_layer(G)

    """ visualize filters """
    visualize_conv_filters(G)


def model_test(G):

    """ set to eval mode """
    G.eval()

    img = np.asarray(imread("../x1.png"))
    img = np.reshape(img, (1, 1, 64, 64))
    img = torch.from_numpy(img)

    """ Transfer data to GPU """

    """ Run Model """
    since = time.time()
    reconimg, cos, sin = G(img)
    print(time.time()-since)

    """ reduce dimension to make images """
    reconimg = torch.squeeze(reconimg)
    img = torch.squeeze(img)
    cos = torch.squeeze(cos)
    sin = torch.squeeze(sin)

    """ torch to numpy """
    reconimg = reconimg.cpu().detach().numpy()
    img = img.cpu().detach().numpy()
    cos = cos.cpu().detach().numpy()
    sin = sin.cpu().detach().numpy()

    """ print images """

    save_reconimg_path = "../test_recon.png"
    imwrite(reconimg, save_reconimg_path)

    reconimg = np.fft.fft2(cos + 1j * sin)
    hologram = np.fft.ifft2(np.multiply(img, np.exp(1j * np.angle(reconimg))))
    reconimg = abs(np.fft.fft2(np.exp(1j * np.angle(hologram))))

    print(np.shape(hologram))
    print(np.shape(reconimg))

    save_gsimg_path = "../test_gs11.png"
    imwrite(reconimg, save_gsimg_path)


if __name__ == "__main__":
    main()
