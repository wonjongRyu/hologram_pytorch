import torch
from torch.autograd import Variable
from utils import *
from data import test_data_loader
from os.path import join
import numpy as np


def test(args, model, epoch):
    """
    In this function:
    1) load dataset
    2) Run model
    3) Print results
    """

    """ Dataset """
    test_loader = test_data_loader(args)
    model.eval()
    volatile = True

    """ Batch Iteration """
    for batch_idx, (image, target) in enumerate(test_loader):

        """ Data """
        if args.is_cuda:
            image, target = image.cuda(), target.cuda()
        image, target = Variable(image, volatile), Variable(target)

        """ Run Model and Get Loss """
        output = model(image)
        output = torch.squeeze(output)
        output = output.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_image_path = join(args.save_image_path, str(i+1)+'_'+str(epoch+1)+'.png')
            imwrite(output[i], save_image_path)
