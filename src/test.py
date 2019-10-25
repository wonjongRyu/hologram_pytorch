import math
from utils import *
from os.path import join


def test(args, model, test_loader, epoch):
    """
    In this function:
    1) load dataset
    2) Run model
    3) Print results
    """

    """ Dataset """
    model.eval()

    """ Batch Iteration """
    for batch_idx, (image, target) in enumerate(test_loader):

        """ Data """
        if args.is_cuda:
            image = image.cuda()

        """ Run Model and Get Loss """
        hologram, output = model(image)
        hologram = torch.squeeze(hologram)
        hologram = hologram.cpu().detach().numpy()
        output = torch.squeeze(output)
        output = output.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            holo = hologram[i]
            h2i = abs(np.fft.fft2(np.exp(1j * holo * 2 * math.pi)))

            save_holo_path = join(args.save_image_path, str(i+1)+'_holo_'+str(epoch+1)+'.png')
            save_recon_path = join(args.save_image_path, str(i+1)+'_recon_'+str(epoch+1)+'.png')
            save_output_path = join(args.save_image_path, str(i+1)+'_output_'+str(epoch+1)+'.png')

            imwrite(h2i, save_recon_path)
            imwrite(holo, save_holo_path)
            imwrite(output[i], save_output_path)
