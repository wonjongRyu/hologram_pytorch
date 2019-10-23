from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from data import train_data_loader
from utils import *
import time
from test import test


def train(args, model):
    """
    In this function:
    1) load dataset
    2) load checkpoint
    3) Training and validation
    4) Visualize results
    """

    """ Dataset """
    train_loader, valid_loader = train_data_loader(args)
    """
    if args.use_preTrain:
        pretrained_resnet_features = get_resnet_features(
            args.is_cuda, train_loader, valid_loader, block_num=18
        )
        train_loader, valid_loader = features_data_loader(
            args, pretrained_resnet_features
        )
    """

    """ Checkpoint """
    """
    if args.resume:
        model, best_acc, epoch_ckpt = ckpt(model)
    """

    """ Training and Validation """
    since = time.time()
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    for epoch in range(0, args.epoch_num+1):
        """ iterate and get loss """
        train_epoch_loss = iteration(args, model, train_loader, phase="train")
        valid_epoch_loss = iteration(args, model, valid_loader, phase="valid")

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        """ print loss """
        if (epoch % args.print_period_error) == 0:
            print_loss(epoch, time.time() - since, train_epoch_loss, valid_epoch_loss)

        """ print image """
        if ((epoch+1) % args.print_period_image) == 0:
            test(args, model, epoch)

    """ Result Graph """
    visualize_graph(train_loss, valid_loss)


def iteration(args, model, data_loader, phase="train", volatile=False):
    """
    In this function:
    1) Phase setting
    2) Define loss function and optimizer
    3) Batch Iteration
    4) Outputs loss
    """

    if phase == "train":
        model.train()
    if phase == "valid":
        model.eval()
        volatile = True

    """ Loss function and Optimizer """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_sum = 0.0

    """ Batch Iteration """
    for batch_idx, (image, target) in enumerate(data_loader):

        """ Data """
        if args.is_cuda:
            image, target = image.cuda(), target.cuda()
        image, target = Variable(image, volatile), Variable(target)

        """ Initialize Gradient """
        if phase == "train":
            optimizer.zero_grad()

        """ Run Model and Get Loss """
        output = model(image)
        loss = criterion(output, target)
        loss_sum += loss.item()

        """ Back Propagation """
        if phase == "train":
            loss.backward()
            optimizer.step()

        """ Clear Memory: Important """
        torch.cuda.empty_cache()

    return loss_sum/len(data_loader)
