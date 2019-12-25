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
        reconimg = torch.squeeze(reconimg)
        #phase = torch.squeeze(phase)

        """ torch to numpy """
        reconimg = reconimg.cpu().detach().numpy()
        #phase = phase.cpu().detach().numpy()
        #img = image.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_reconimg_path = args.save_path_of_images+'/img'+str(i+1)+'_'+str(epoch)+'_R.png'
            #save_gsimg_path = args.save_path_of_images+'/img'+str(i+1)+'_'+str(epoch)+'_GS1.png'
            #gsimg = normalize_img(gs1time(img[i], phase[i]))
            imwrite(reconimg[i], save_reconimg_path)
            #imwrite(gsimg, save_gsimg_path)



def test_sincos(args, G, test_loader, epoch):
    """ test function """

    """ set to eval mode """
    G.eval()

    """ Batch Iteration """
    for batch_idx, image in enumerate(test_loader):

        """ Transfer data to GPU """
        if args.is_cuda_available:
            image = image.cuda()

        """ Run Model """
        reconimg, cos, sin = G(image)

        """ reduce dimension to make images """
        reconimg = torch.squeeze(reconimg)
        cos = torch.squeeze(cos)
        sin = torch.squeeze(sin)

        """ torch to numpy """
        reconimg = reconimg.cpu().detach().numpy()
        img = image.cpu().detach().numpy()
        cos = cos.cpu().detach().numpy()
        sin = sin.cpu().detach().numpy()

        """ print images """
        for i in range(len(test_loader.dataset)):
            save_reconimg_path = args.save_path_of_images+'/img'+str(i+1)+'_'+str(epoch)+'_R.png'
            save_gsimg_path = args.save_path_of_images+'/img'+str(i+1)+'_'+str(epoch)+'_GS1.png'
            gsimg = gs1cossin(img[i], cos[i], sin[i])
            imwrite(reconimg[i], save_reconimg_path)
            imwrite(gsimg, save_gsimg_path)
