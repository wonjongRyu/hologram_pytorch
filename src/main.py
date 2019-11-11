import argparse
from models.HGN import HGN_GAN
from train import train
from test import test
from torchsummary import summary
from data import *
from ops import netD


def parse_args():
    """ parser_args function """

    parser = argparse.ArgumentParser(description="Hologram Generation Net")

    """ Dataset Path """
    parser.add_argument("--path_of_dataset", type=str, default="../dataset_test")

    """ Training Condition """
    parser.add_argument("--use_preTrained_model", type=int, default=False)
    parser.add_argument("--is_cuda_available", type=int, default=True)
    parser.add_argument("--list_of_block_numbers", type=list, default=[2, 2, 2, 2, 2])
    parser.add_argument("--epoch_max", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)

    """ Learning Rate """
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--decay_coefficient_of_learning_rate", type=float, default=3)
    parser.add_argument("--decay_cycle_of_learning_rate", type=int, default=500)

    """ Loss Ratio """
    parser.add_argument("--loss_ratio", type=float, default=1)
    parser.add_argument("--change_cycle_of_loss_ratio", type=float, default=500)

    """ Print Cycles """
    parser.add_argument("--print_cycle_of_loss", type=int, default=10)
    parser.add_argument("--print_cycle_of_image", type=int, default=100)

    """ Save Paths """
    parser.add_argument("--save_path_of_outputs", type=str, default="../outputs")
    parser.add_argument("--save_path_of_image", type=str, default="../outputs/images")
    parser.add_argument("--save_path_of_model", type=str, default="../outputs/models")
    parser.add_argument("--save_path_of_loss", type=str, default="../outputs/loss")

    return check_args(parser.parse_args())


def main():
    """ Main function """

    """ Load Arguments """
    args = parse_args()

    """ Check GPU """
    args.is_cuda_available = torch.cuda.is_available()

    """ Define Network """
    G = HGN_GAN(args.list_of_block_numbers)
    G.load_state_dict(torch.load("../models/GANfc.pt"))
    D = netD()

    if args.is_cuda_available:
        G.cuda()
        D.cuda()

    """ check parameter """
    # summary(model, (1, 64, 64))

    """ Train model """
    train(args, G, D)

    """ save model """
    # torch.save(G.state_dict(), "../models/GANfc.pt")


if __name__ == "__main__":
    main()
