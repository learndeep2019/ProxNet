from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pickle as pk
import numpy as np

class Sketchy(Dataset):
    '''
    Customized Dataset class, which will read the data from pickle files.

    Args:
        file_path (str): Path to the pickle file.
        train (bool): Training dataset if it is true, otherwise testing dataset. Default True.
        transform1: Optional transform to be applied on a sample from view1.
        transform2: Optional transform to be applied on a sample from view2.

    Returns:
        Objects (Dataset): sample of this dataset will be a tuple (img_v1, img_v2, label).
    '''
    def __init__(self, file_path='./data/sketch100.pk', train=True, transform1=None, transform2=None,):
        with open(file_path, 'rb') as f:
            imgs = pk.load(f)

        keys = 'train' if train else 'test'
        self.view1 = imgs['v1'][keys]
        self.view2 = imgs['v2'][keys]
        self.label = imgs['label'][keys].astype(np.int64)

        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, idx):
        img1, img2, target = self.view1[idx], self.view2[idx], self.label[idx]

        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)

        if self.transform1 is not None:
            img1 = self.transform1(img1)
        if self.transform2 is not None:
            img2 = self.transform2(img2)

        return img1, img2, target

    def __len__(self):
        return len(self.label)

def get_loader(config):
    '''
    build dataloader for training and testing dataset.

    Args:
        config (dict): configuration dictionary.

    Returns:
        train_loader (Dataloader): Dataloader for training dataset.
        test_loader (Dataloader): Dataloader for testing dataset.
    '''
    file_path = config['file_path']
    batch_size = config['batch_size']

    transformer1 = transforms.Compose([transforms.RandomCrop(256, 25),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.418, 0.462, 0.464],[0.284, 0.259, 0.269])
                    ])
    transformer2 = transforms.Compose([transforms.RandomCrop(256, 25),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.958, 0.958, 0.958],[0.196, 0.196, 0.196])
                    ])
    transformer1_ = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.418, 0.462, 0.464],[0.284, 0.259, 0.269])
                    ])
    transformer2_ = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize([0.958, 0.958, 0.958],[0.196, 0.196, 0.196])
                    ])
    train = Sketchy(file_path=file_path, transform1=transformer1, transform2=transformer2)
    test = Sketchy(file_path=file_path, train=False, transform1=transformer1_, transform2=transformer2_)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size)
    return train_loader, test_loader

if __name__ == "__main__":
    sk = Sketchy()