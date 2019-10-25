from os.path import join
from glob import glob
from utils import *
from torch.utils.data import Dataset, DataLoader


class myData(Dataset):
    def __init__(self, root_dir):
        self.files = glob(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = np.asarray(imread(self.files[idx]))
        target = gs_algorithm(image, 100)

        image = np.reshape(image, (64, 64, 1))
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)

        target = np.reshape(target, (64, 64, 1))
        target = np.swapaxes(target, 0, 2)
        target = np.swapaxes(target, 1, 2)

        return image, target.astype(np.float32)


def data_loader(args):
    train_images = myData(join(args.dataset_path, "train/*.*"))
    valid_images = myData(join(args.dataset_path, "valid/*.*"))
    test_images = myData(join(args.dataset_path, "test/*.*"))
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size=args.batch_size, shuffle=False)  # ***FALSE***

    return train_loader, valid_loader, test_loader
