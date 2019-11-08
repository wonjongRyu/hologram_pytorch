import argparse
from models.HGN import HGN
from train import train
from test import test
from torchsummary import summary
from data import *
from ops import netD


def parse_args():
    """ parser_args function """

    parser = argparse.ArgumentParser(description="Hologram Generation Net")

    """ Dataset """
    parser.add_argument("--dataset_path", type=str, default="../dataset4000")
    parser.add_argument("--use_preTrain", type=int, default=False)

    """ Training Condition """
    parser.add_argument("--is_cuda", type=int, default=True)
    parser.add_argument("--block_num", type=list, default=[2, 2, 2, 2, 2])
    parser.add_argument("--epoch_max", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--lr_decay_param", type=float, default=3)
    parser.add_argument("--lr_decay_period", type=int, default=500)
    parser.add_argument("--loss_ratio", type=float, default=1)
    parser.add_argument("--change_loss_ratio_at", type=float, default=500)

    """ Results """
    parser.add_argument("--print_period_error", type=int, default=10)
    parser.add_argument("--print_period_image", type=int, default=100)

    """ Directories """
    parser.add_argument("--save_image_path", type=str, default="../images")
    parser.add_argument("--save_loss_path", type=str, default="../loss")
    parser.add_argument("--save_model_path", type=str, default="../models")
    return check_args(parser.parse_args())


def main():
    """ Main function """

    """ Load Arguments """
    args = parse_args()

    """ Check GPU """
    args.is_cuda = torch.cuda.is_available()

    """ Define Network """
    G = HGN(args.block_num)
    G.load_state_dict(torch.load("../models/GANfc.pt"))
    D = netD()

    if args.is_cuda:
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
