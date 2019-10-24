from torch.autograd import Variable
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
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    for epoch in range(0, args.epoch_num+1):
        """ run 1 epoch and get loss """
        train_epoch_loss = iteration(args, model, train_loader, phase="train")
        valid_epoch_loss = iteration(args, model, valid_loader, phase="valid")

        """ Log loss """
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        """ Print loss """
        if (epoch % args.print_period_error) == 0:
            print_loss(epoch, time.time() - since, train_epoch_loss, valid_epoch_loss)

        """ Print image """
        if ((epoch+1) % args.print_period_image) == 0:
            test(args, model, test_loader, epoch)

    """ Visualize results """
    visualize_graph(train_loss, valid_loss)

    print('=================[ train finish ]=================')


def iteration(args, model, data_loader, phase="train", volatile=False):
    """ iteration function """

    """ Phase setting: train or valid """
    if phase == "train":
        model.train()
    if phase == "valid":
        model.eval()
        volatile = True

    """ Define loss function and optimizer """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_sum = 0.0

    """ Start batch iteration """
    for batch_idx, (image, target) in enumerate(data_loader):

        """ Load data """
        if args.is_cuda:
            image, target = image.cuda(), target.cuda()
        image, target = Variable(image, volatile), Variable(target)

        """ Initialize gradient """
        if phase == "train":
            optimizer.zero_grad()

        """ Run model and calculate loss """
        output = model(image)
        loss = criterion(output, target)
        loss_sum += loss.item()

        """ Back propagation """
        if phase == "train":
            loss.backward()
            optimizer.step()

        """ Clear memory """
        torch.cuda.empty_cache()

    return loss_sum/len(data_loader)
