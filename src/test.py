from utils import *
from os.path import join


def test(args, model, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    model.eval()

    """ Batch Iteration """
    for batch_idx, (image, hologram) in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda_available:
            image, hologram = image.cuda(), hologram.cuda()

        """ Run Model """
        generate_hologram = model(image)

        """ reduce dimension to make images """
        generate_hologram = torch.squeeze(generate_hologram)
        # reconimg = torch.squeeze(reconimg)

        """ torch to numpy """
        generate_hologram = generate_hologram.cpu().detach().numpy()
        # reconimg = reconimg.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_holo_path = args.save_path_of_images + '/' + str(i + 1) + '_H_' + str(epoch) + '.png'
            save_reconimg_path = args.save_path_of_images + '/' + str(i+1)+'_R_'+str(epoch)+'.png'

            reconimg = abs(np.fft.fft2(np.exp(1j * generate_hologram[i])))
            imwrite(generate_hologram[i], save_holo_path)
            imwrite(reconimg, save_reconimg_path)
