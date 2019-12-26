from os.path import join
from utils import *
from torch.utils.data import Dataset, DataLoader


class myDataset1(Dataset):
    def __init__(self, root_dir):
        self.images = glob(join(root_dir, "*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = np.asarray(imread(self.images[idx]))
        images = np.reshape(images, (128, 128, 1))
        images = np.swapaxes(images, 0, 2)
        images = np.swapaxes(images, 1, 2)

        return images


class myDataset2(Dataset):
    def __init__(self, root_dir):
        self.images = glob(join(root_dir, "images/*.*"))
        self.target = glob(join(root_dir, "holograms/*.*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = np.asarray(imread(self.images[idx]))
        target = np.asarray(imread(self.target[idx]))

        images = np.reshape(images, (64, 64, 1))
        images = np.swapaxes(images, 0, 2)
        images = np.swapaxes(images, 1, 2)

        target = np.reshape(target, (64, 64, 1))
        target = np.swapaxes(target, 0, 2)
        target = np.swapaxes(target, 1, 2)

        return images, target


class myDataset2Input(Dataset):
    def __init__(self, root_dir):
        self.images = glob(join(root_dir, "images/*.*"))
        self.holo1 = glob(join(root_dir, "gs1/*.*"))
        self.holo100 = glob(join(root_dir, "gs100/*.*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = np.asarray(imread(self.images[idx]))
        holo1 = np.asarray(imread(self.holo1[idx]))
        holo100 = np.asarray(imread(self.holo100[idx]))

        images = np.reshape(images, (64, 64, 1))
        images = np.swapaxes(images, 0, 2)
        images = np.swapaxes(images, 1, 2)

        holo1 = np.reshape(holo1, (64, 64, 1))
        holo1 = np.swapaxes(holo1, 0, 2)
        holo1 = np.swapaxes(holo1, 1, 2)

        holo100 = np.reshape(holo100, (64, 64, 1))
        holo100 = np.swapaxes(holo100, 0, 2)
        holo100 = np.swapaxes(holo100, 1, 2)

        return images, holo1, holo100


def data_loader1(args):
    """ Data Loader"""

    """ Load image data """
    train_images = myDataset1(join(args.path_of_dataset, "train"))
    valid_images = myDataset1(join(args.path_of_dataset, "valid"))
    test_images = myDataset1(join(args.path_of_dataset, "test"))

    """ Wrap them with DataLoader structure """
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size=args.batch_size, shuffle=False)  # ***FALSE***

    return train_loader, valid_loader, test_loader


def data_loader2(args):
    """ Data Loader"""

    """ Load image data """
    train_images = myDataset2(join(args.path_of_dataset, "train"))
    valid_images = myDataset2(join(args.path_of_dataset, "valid"))
    test_images = myDataset2(join(args.path_of_dataset, "test"))

    """ Wrap them with DataLoader structure """
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size=args.batch_size, shuffle=False)  # ***FALSE***

    return train_loader, valid_loader, test_loader
