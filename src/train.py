import torch.nn as nn
import torch.optim as optim
from utils import *
import time
from test import test
from data import data_loader1, data_loader2


def train(args, G, D):
    """ train function """
    """ Print start time """
    print_start_time()

    """ Make output directories """
    make_output_folders(args)

    """ Train model and validate it """
    since = time.time()

    """ Start iteration """
    for epoch in range(1, args.epoch_max+1):

        """ run 1 epoch and get loss """
        train_loader, valid_loader, test_loader = data_loader1(args)
        train_loss = iteration_GAN(args, G, D, train_loader, phase="train")
        valid_loss = iteration_GAN(args, G, D, valid_loader, phase="valid")

        """ Print loss """
        if (epoch % args.print_cycle_of_loss) == 0:
            print_loss(epoch, time.time()-since, train_loss, valid_loss)
            # record_loss(args, epoch, time.time()-since, train_loss, valid_loss)

        """ Print image """
        if (epoch % args.print_cycle_of_image) == 0:
            test(args, G, test_loader, epoch)
            # visualize_conv_layer(epoch, model)

        """ Change the ratio of losses """
        # if epoch == args.change_cycle_of_loss_ratio:
        #    args.loss_ratio = 0

        """ Decay Learning Rate """
        if (epoch % args.decay_cycle_of_learning_rate) == 0:
            args.learning_rate = args.learning_rate/args.decay_coefficient_of_learning_rate

    print('======================[ train finished ]======================')


def iteration_CGH(args, G, data_loader, phase="train"):
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
        loss_sum_holo = 0.0
        loss_sum_image = 0.0

        """ Start batch iteration """
        for batch_idx, (image, holo) in enumerate(data_loader):

            """ Transfer data to GPU """
            if args.is_cuda_available:
                image, holo = image.cuda(), holo.cuda()

            """ Run model """
            hologram, reconimg = G(image)

            """ Calculate batch loss """
            loss_holo = criterion(hologram, holo)
            loss_image = criterion(reconimg, image)

            """ Add to get epoch loss """
            loss_sum_holo += loss_holo.item()
            loss_sum_image += loss_image.item()  # 여기 item() 없으면 GPU 박살

            """ Back propagation """
            if phase == "train":
                optimizer.zero_grad()
                loss = args.loss_ratio * loss_holo + (1 - args.loss_ratio) * loss_image
                loss.backward()
                optimizer.step()

            """ Clear memory """
            torch.cuda.empty_cache()

        return loss_sum_holo / len(data_loader)

        # return loss_sum_holo / len(data_loader), loss_sum_image / len(data_loader)


def iteration_GAN(args, G, D, data_loader, phase="train"):
    """ iteration function """

    """ Phase setting: train or valid """
    if phase == "train":
        G.train()
        D.train()
    if phase == "valid":
        G.eval()
        D.eval()

    """ Define loss function and optimizer """
    criterion_MSE = nn.MSELoss()
    criterion_BCE = nn.BCELoss()
    optimizerD = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    """ Initialize the loss_sum """
    I_loss_sum = 0.0
    """
    D_real_sum = 0.0
    D_fake_sum = 0.0
    I_BCE_sum = 0.0
    I_MSE_sum = 0.0
    """

    """ Start batch iteration """
    for batch_idx, image in enumerate(data_loader):

        """ Make labels """
        real_label, fake_label = make_labels(image.size(0))

        """ Transfer data to GPU """
        if args.is_cuda_available:
            image, real_label, fake_label = image.cuda(), real_label.cuda(), fake_label.cuda()

        """ Run model """
        _, reconimg = G(image)

        """ Discriminator """
        D_real_logits = D(image)
        D_fake_logits = D(reconimg.detach())

        """ D Loss """
        D_real_loss = criterion_BCE(D_real_logits, real_label)
        D_fake_loss = criterion_BCE(D_fake_logits, fake_label)

        """ G Loss """
        I_BCE_loss = criterion_BCE(D_fake_logits, real_label)
        I_MSE_loss = criterion_MSE(reconimg, image)

        """ BP """
        if phase == "train":
            optimizerD.zero_grad()
            D_real_loss.backward()
            D_fake_loss.backward(retain_graph=True)
            optimizerD.step()

            optimizerG.zero_grad()
            G_loss = 0.9*I_BCE_loss + 0.1*I_MSE_loss
            G_loss.backward()
            # I_BCE_loss.backward()
            # I_MSE_loss.backward()
            optimizerG.step()

        """ Add to get epoch loss """
        I_loss_sum += I_MSE_loss.item()    # 여기 item() 없으면 GPU

        """
        D_real_sum += D_real_loss.item()
        D_fake_sum += D_fake_loss.item()
        I_BCE_sum += I_BCE_loss.item()
        I_MSE_sum += I_MSE_loss.item()
        """

        """ Clear memory """
        torch.cuda.empty_cache()

    # return D_real_sum/len(data_loader), D_fake_sum/len(data_loader), I_BCE_sum/len(data_loader), I_MSE_sum/len(data_loader)

    return I_loss_sum/len(data_loader)
