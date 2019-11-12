import torch.nn as nn
import torch.optim as optim
from utils import *
import time
from test import test
from data import data_loader2


def train_CGH(args, G):
    """ train function """

    print_start_time()
    make_output_folders(args)
    since = time.time()

    """ Start iteration """
    for epoch in range(1, args.epoch_max+1):

        """ run 1 epoch and get loss """
        train_loader, valid_loader, test_loader = data_loader2(args)
        train_loss = iteration(args, G, train_loader, phase="train")
        valid_loss = iteration(args, G, valid_loader, phase="valid")

        """ Print loss """
        if (epoch % args.print_cycle_of_loss) == 0:
            print_loss(epoch, time.time()-since, train_loss, valid_loss)
            record_on_csv(args, epoch, time.time()-since, train_loss, valid_loss)

        """ Print image """
        if (epoch % args.print_cycle_of_images) == 0:
            test(args, G, test_loader, epoch)
            # visualize_conv_layer(epoch, model)

        """ Change the ratio of losses """
        # if epoch == args.change_cycle_of_loss_ratio:
        #    args.loss_ratio = 0

        """ Decay Learning Rate """
        # if (epoch % args.decay_cycle_of_learning_rate) == 0:
        #     args.learning_rate = args.learning_rate/args.decay_coefficient_of_learning_rate

    print('======================[ train finished ]======================')


def iteration(args, G, data_loader, phase):
        """ iteration function """

        """ Phase setting: train or valid """
        if phase == "train":
            G.train()
        if phase == "valid":
            G.eval()

        """ Define loss function and optimizer """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(G.parameters(), lr=args.learning_rate)

        """ Initialize the loss_sum """
        loss_sum_hologram = 0.0
        # loss_sum_image = 0.0

        """ Start batch iteration """
        for batch_idx, (image, hologram) in enumerate(data_loader):

            """ Transfer data to GPU """
            if args.is_cuda_available:
                image, hologram = image.cuda(), hologram.cuda()

            """ Run model """
            generated_hologram = G(image)

            """ Calculate batch loss """
            loss_hologram = criterion(generated_hologram, hologram)
            # loss_image = criterion(reconimg, image)

            """ Back propagation """
            if phase == "train":
                optimizer.zero_grad()
                # loss = args.loss_ratio * loss_hologram + (1 - args.loss_ratio) * loss_image
                loss_hologram.backward()
                optimizer.step()

            """ Add to get epoch loss """
            loss_sum_hologram += loss_hologram.item()
            # loss_sum_image += loss_image.item()  # 여기 item() 없으면 GPU 박살

            """ Clear memory """
            torch.cuda.empty_cache()

        return loss_sum_hologram / len(data_loader)

        # return loss_sum_holo / len(data_loader), loss_sum_image / len(data_loader)
