from os.path import join
from utils import *
from torch.utils.data import Dataset, DataLoader


class myDataset(Dataset):
    def __init__(self, args, mode):
        self.images = glob(join(args.path_of_dataset, mode, "*.png"))
        self.csv_path = args.path_of_dataset + '/' + mode + '.csv'
        self.sz = args.img_size

        with open(self.csv_path, 'r') as f:
            reader = csv.reader(f)
            norm_arr = list(reader)
        norm_arr = np.squeeze(norm_arr)
        norm_arr = list(map(float, norm_arr))
        self.norm_arr = np.asarray(norm_arr)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = np.asarray(imread(self.images[idx]))
        images = np.reshape(images, (self.sz, self.sz, 1))
        images = np.swapaxes(images, 0, 2)
        images = np.swapaxes(images, 1, 2)
        norm = self.norm_arr[idx]

        return images, norm


def data_loader(args):
    """ Data Loader"""

    """ Load image data """
    train_images = myDataset(args, "train")
    valid_images = myDataset(args, "valid")
    test_images = myDataset(args, "test")

    """ Wrap them with DataLoader structure """
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size=args.batch_size, shuffle=False)  # ***FALSE***

    return train_loader, valid_loader, test_loader
