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
        train_loss1, train_loss2, train_total = iteration(since, args, model, epoch, train_loader, phase="train")
        valid_loss1, valid_loss2, valid_total = iteration(since, args, model, epoch, valid_loader, phase="valid")
        """ Log loss """
        train_loss.append(train_total)
        valid_loss.append(valid_total)
        """ Print loss """
        if (epoch % args.print_period_error) == 0:
            print_2_loss(epoch, time.time() - since, train_loss1, train_loss2, train_total, valid_loss1, valid_loss2, valid_total)
        """ Print image """
        if ((epoch+1) % args.print_period_image) == 0:
            test(args, model, test_loader, epoch)
        # if epoch == 100:
        #     args.loss_ratio = 0

    """ Visualize results """
    visualize_graph(train_loss, valid_loss)

    print('=================[ train finish ]=================')


def iteration(since, args, model, epoch, data_loader, phase="train"):
    """ iteration function """

    """ Phase setting: train or valid """
    if phase == "train":
        model.train()
    if phase == "valid":
        model.eval()

    """ Define loss function and optimizer """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss1_sum = 0.0
    loss2_sum = 0.0
    total_sum = 0.0

    """ Start batch iteration """
    for batch_idx, (image, target) in enumerate(data_loader):
        """ Load data """
        if args.is_cuda:
            image, target = image.cuda(), target.cuda()
        """ Initialize gradient """
        if phase == "train":
            optimizer.zero_grad()

        """ Run model and calculate loss """
        hologram, reconimg = model(image)
        if epoch < 50:
            total = criterion(hologram, target)
        else:
            total = criterion(reconimg, image)
        """ 
        loss1 = criterion(hologram, target)
        loss2 = criterion(reconimg, image)
        total = args.loss_ratio*loss1+(1-args.loss_ratio)*loss2

        loss1_sum += loss1.item()
        loss2_sum += loss2.item()
        """
        total_sum += total.item()  # 여기 item() 없으면 GPU 박살

        """ Back propagation """
        if phase == "train":
            total.backward()
            optimizer.step()
        """ Clear memory """
        torch.cuda.empty_cache()

    return loss1_sum/len(data_loader), loss2_sum/len(data_loader), total_sum/len(data_loader)
