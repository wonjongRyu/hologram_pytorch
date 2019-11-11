from utils import *
from os.path import join


def test(args, model, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    model.eval()

    """ Batch Iteration """
    for batch_idx, image in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda:
            image = image.cuda()

        """ Run Model """
        hologram, reconimg = model(image)

        """ reduce dimension to make images """
        # hologram = torch.squeeze(hologram)
        reconimg = torch.squeeze(reconimg)

        """ torch to numpy """
        # hologram = hologram.cpu().detach().numpy()
        reconimg = reconimg.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_holo_path = join(args.save_path_of_image, str(i + 1) + '_holo_' + str(epoch) + '.png')
            save_output_path = join(args.save_path_of_image, str(i+1)+'_output_'+str(epoch)+'.png')
            imwrite(hologram[i], save_holo_path)
            imwrite(reconimg[i], save_output_path)
