import torch.nn as nn
import torch.optim as optim
from utils import *
import time
from test import test


def train(args, model, train_loader, valid_loader, test_loader):
    """ train function """

    """ Print start time """
    print_start_time()

    """ Train model and validate it """
    since = time.time()
    for epoch in range(1, args.epoch_num+1):
        """ run 1 epoch and get loss """
        train_loss = iteration(args, model, train_loader, phase="train")
        valid_loss = iteration(args, model, valid_loader, phase="valid")

        """ Print loss """
        if (epoch % args.print_period_error) == 0:
            print_loss(epoch, time.time()-since, train_loss, valid_loss)
            record_loss(args, epoch, time.time()-since, train_loss, valid_loss)

        """ Print image """
        if (epoch % args.print_period_image) == 0:
            test(args, model, test_loader, epoch)
            visualize_conv_layer(epoch, model)

        """ Change the ratio of losses """
        if epoch == args.change_loss_ratio_at:
            args.loss_ratio = 0

        """ Decay Learning Rate """
        if (epoch % args.lr_decay_period) == 0:
            args.learning_rate = args.learning_rate/args.lr_decay_param

    print('======================[ train finished ]======================')


def iteration(args, model, data_loader, phase="train"):
    """ iteration function """

    """ Phase setting: train or valid """
    if phase == "train":
        model.train()
    if phase == "valid":
        model.eval()

    """ Define loss function and optimizer """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    """ Initialize the loss_sum """
    loss_sum_holo = 0.0
    loss_sum_image = 0.0

    """ Start batch iteration """
    for batch_idx, (image, holo) in enumerate(data_loader):

        """ Transfer data to GPU """
        if args.is_cuda:
            image, holo = image.cuda(), holo.cuda()

        """ Run model """
        hologram, reconimg = model(image)

        """ Calculate batch loss """
        loss_holo = criterion(hologram, holo)
        loss_image = criterion(reconimg, image)

        """ Add to get epoch loss """
        loss_sum_holo += loss_holo.item()
        loss_sum_image += loss_image.item()  # 여기 item() 없으면 GPU 박살

        """ Back propagation """
        if phase == "train":
            optimizer.zero_grad()
            loss = args.loss_ratio*loss_holo+(1-args.loss_ratio)*loss_image
            loss.backward()
            optimizer.step()

        """ Clear memory """
        torch.cuda.empty_cache()

    return loss_sum_holo/len(data_loader), loss_sum_image/len(data_loader)
