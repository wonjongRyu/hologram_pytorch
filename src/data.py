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
        image = np.reshape(image, (64, 64, 1))
        image = np.swapaxes(image, 0, 2)
        image = np.swapaxes(image, 1, 2)
        target = image
        # target = target.astype(np.float32)
        return image, target


def train_data_loader(args):
    train_images = myData(join(args.dataset_path, "train/*.*"))
    valid_images = myData(join(args.dataset_path, "valid/*.*"))
    train_loader = DataLoader(train_images, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_images, batch_size=args.batch_size, shuffle=True)

    return train_loader, valid_loader


def test_data_loader(args):
    test_images = myData(join(args.dataset_path, "test/*.*"))
    test_loader = DataLoader(test_images, batch_size=args.batch_size, shuffle=False)  # ***FALSE***

    return test_loader


class featuresData(Dataset):
    def __init__(self, feat, labels):
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)

    def __getitem__(self, idx):
        return self.conv_feat[idx], self.labels[idx]


def features_data_loader(args, pretrained_resnet_features):
    [
        train_features,
        train_labels,
        valid_features,
        valid_labels,
    ] = pretrained_resnet_features
    train_features = featuresData(train_features, train_labels)
    valid_features = featuresData(valid_features, valid_labels)
    train_loader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_features, batch_size=args.batch_size, shuffle=True)
    return train_loader, valid_loader
