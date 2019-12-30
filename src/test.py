from utils import *
from os.path import join
from ops import *


def test(args, G, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    G.eval()

    """ Batch Iteration """
    for batch_idx, (image, _) in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda_available:
            image = image.cuda()

        """ Run Model """
        hologram, reconimg = G(image)

        """ reduce dimension to make images """
        reconimg = torch.squeeze(reconimg)
        hologram = torch.squeeze(hologram)

        """ GPU2CPU, Torch2Numpy """
        reconimg = reconimg.cpu().detach().numpy()
        hologram = hologram.cpu().detach().numpy()

        """ save tensor """
        save_tensor(args, hologram)

        """ save images """
        for i in range(len(test_loader.dataset)):
            normalized_img = imnorm(reconimg[i])
            file_path = combine('img', i+1, '_', epoch, '.png')
            save_path = os.path.join(args.save_path_of_images, file_path)
            imwrite(normalized_img, save_path)
