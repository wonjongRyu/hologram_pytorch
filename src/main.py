import argparse
from models.HGN import HGN
from train_img import train_img
# from torchsummary import summary
from data import *
from utils import *


def parse_args():
    """ parser_args function """

    parser = argparse.ArgumentParser(description="Hologram Generation Net")

    """ Training """
    parser.add_argument("--description", type=str, default='Nothing changed')

    """ Dataset Path """
    parser.add_argument("--since", type=str, default='YearMonthDay_HourMinutes')
    parser.add_argument("--path_of_dataset", type=str, default="../dataset/object16000_64")
    parser.add_argument("--img_size", type=int, default=64)

    """ Training Condition """
    parser.add_argument("--use_preTrained_model", type=int, default=False)
    parser.add_argument("--is_cuda_available", type=int, default=True)

    """ Hyperparameters: Architecture """
    parser.add_argument("--block_nums", type=list, default=[2, 2, 2, 2, 2])
    parser.add_argument("--epoch_max", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--channel_size", type=int, default=32)

    """ Hyperparameters: Learning Rate """
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--decay_coeff_of_learning_rate", type=float, default=3)
    parser.add_argument("--decay_cycle_of_learning_rate", type=int, default=500)

    """ Hyperparameters: Loss Ratio """
    parser.add_argument("--loss_ratio", type=float, default=1)
    parser.add_argument("--change_cycle_of_loss_ratio", type=float, default=500)

    """ Print Cycles """
    parser.add_argument("--save_cycle_of_loss",   type=int, default=1)
    parser.add_argument("--save_cycle_of_images", type=int, default=1)
    parser.add_argument("--save_cycle_of_models", type=int, default=500)

    """ Save Paths """
    parser.add_argument("--save_path_of_outputs", type=str, default="../outputs")
    parser.add_argument("--save_path_of_images", type=str, default="../outputs/images")
    parser.add_argument("--save_path_of_models", type=str, default="../outputs/models")
    parser.add_argument("--save_path_of_layers", type=str, default="../outputs/layers")
    parser.add_argument("--save_path_of_tensor", type=str, default="../outputs/tensor")
    parser.add_argument("--save_path_of_loss", type=str, default="../outputs/loss.csv")
    parser.add_argument("--save_path_of_arch", type=str, default="../outputs/arch.csv")
    parser.add_argument("--save_path_of_args", type=str, default="../outputs/args.csv")

    return parser.parse_args()


def main():

    """ Load Arguments """
    args = parse_args()
    args.description = 'cgh prepared for ground truth'

    """ initialization """
    save_start_time(args)
    make_csvfile_and_folders(args)

    """ Define Network """
    # G = DenseNetBC_100_12()
    G = HGN(args)
    # print(G)
    # D = netD()

    """ Load Models """
    # G.load_state_dict(torch.load("../models/GANfc.pt"))

    """ Check GPU """
    args.is_cuda_available = torch.cuda.is_available()
    if args.is_cuda_available:
        G.cuda()
        # D.cuda()

    """ check parameter """
    summary(args, G, (1, args.img_size, args.img_size))

    """ Train model """
    train_img(args, G)

    """ save model """
    # torch.save(G.state_dict(), "../models/HGN_sincos_loss_finished.pt")


if __name__ == "__main__":
    main()
