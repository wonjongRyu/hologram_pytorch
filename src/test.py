from utils import *
from os.path import join


def test(args, model, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    model.eval()

    """ Batch Iteration """
    for batch_idx, image in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda_available:
            image = image.cuda()

        """ Run Model """
        reconimg = model(image)

        """ reduce dimension to make images """
        reconimg = torch.squeeze(reconimg)

        """ torch to numpy """
        reconimg = reconimg.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_reconimg_path = args.save_path_of_images+'/R'+str(i+1)+'_'+str(epoch)+'.png'
            imwrite(reconimg[i], save_reconimg_path)
