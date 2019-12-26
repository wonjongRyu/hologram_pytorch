from utils import *
from os.path import join


def test(args, G, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    G.eval()

    """ Batch Iteration """
    for batch_idx, image in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda_available:
            image = image.cuda()

        """ Run Model """
        phase, reconimg = G(image)

        """ reduce dimension to make images """
        phase = torch.squeeze(phase)
        reconimg = torch.squeeze(reconimg)

        """ torch to numpy """
        phase = phase.cpu().detach().numpy()
        reconimg = reconimg.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_reconimg_path = args.save_path_of_images+'/img'+str(i+1)+'_'+str(epoch)+'.png'
            imwrite(reconimg[i], save_reconimg_path)