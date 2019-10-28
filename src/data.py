from os.path import join
from glob import glob
from utils import *
from torch.utils.data import Dataset, DataLoader


class myData(Dataset):
    def __init__(self, root_dir):
        self.images = glob(join(root_dir, "images/*.*"))
        self.holograms = glob(join(root_dir, "holograms/*.*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = np.asarray(imread(self.images[idx]))
        holograms = np.asarray(imread(self.holograms[idx]))

        images = np.reshape(images, (64, 64, 1))
        images = np.swapaxes(images, 0, 2)
        images = np.swapaxes(images, 1, 2)

        holograms = np.reshape(holograms, (64, 64, 1))
        holograms = np.swapaxes(holograms, 0, 2)
        holograms = np.swapaxes(holograms, 1, 2)

        return images, holograms


def data_loader(args):
    train_images = myData(join(args.dataset_path, "train"))
    valid_images = myData(join(args.dataset_path, "valid"))
    test_images = myData(join(args.dataset_path, "test"))
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size=args.batch_size, shuffle=False)  # ***FALSE***

    return train_loader, valid_loader, test_loader
