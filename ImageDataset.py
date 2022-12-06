import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    if I.mode is not 'RGB':
        I = I.convert('RGB')
    return I


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 transform=None,
                 test=False,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = pd.read_csv(csv_file, sep='\t', header=None)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.test = test
        self.transform = transform
        self.loader = get_loader()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        if self.test:
            image_name = os.path.join(self.img_dir, self.data.iloc[index, 0])
            I = self.loader(image_name)
            if self.transform is not None:
                I = self.transform(I)

            mos = self.data.iloc[index, 1]
            std = self.data.iloc[index, 2]
            sample = {'I': I, 'mos': mos, 'std': std, 'path': image_name}
        else:
            image_name1 = os.path.join(self.img_dir, self.data.iloc[index, 0])
            image_name2 = os.path.join(self.img_dir, self.data.iloc[index, 1])

            I1 = self.loader(image_name1)
            I2 = self.loader(image_name2)
            y = torch.FloatTensor(self.data.iloc[index, 2:].tolist())
            if self.transform is not None:
                I1 = self.transform(I1)
                I2 = self.transform(I2)

            sample = {'I1': I1, 'I2': I2, 'y': y[0], 'std1': y[1], 'std2': y[2], 'yb': y[3], 'num': y[4]}

        return sample

    def __len__(self):
        return len(self.data.index)


class ImageDataset2(Dataset):
    def __init__(self, csv_file_list,
                 img_dir_list,
                 transform=None,
                 transform2=None,
                 max_train_sample=8125, # kadid10k
                 expansion=3,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        print('start loading csv data...')
        self.data = []
        self.max_train_sample = max_train_sample
        for i, item in enumerate(csv_file_list):
            data_item = pd.read_csv(item, sep='\t', header=None)
            #self.data.append()
            db_size = len(data_item)
            repeats = int(np.ceil(self.max_train_sample / db_size) - 1)
            offset = (self.max_train_sample % db_size)
            original_data = data_item
            if repeats > 1:
                for j in range(repeats - 1):
                    data_item = pd.concat([data_item, original_data])
            if offset > 0:
                offset_data = original_data.iloc[:offset, :]
                data_item = pd.concat([data_item, offset_data])
            for _ in range(expansion-1):
                data_item = pd.concat([data_item, data_item])
            self.data.append(data_item)

        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir_list = img_dir_list
        self.transform = transform
        self.transform2 = transform2
        self.loader = get_loader()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        sample = []
        for i, item in enumerate(self.data):
            image_name = os.path.join(self.img_dir_list[i], item.iloc[index, 0])
            I = self.loader(image_name)
            if (i == 4) | (i == 6):
                I = self.transform2(I)
            else:
                I = self.transform(I)

            mos = item.iloc[index, 1]
            std = item.iloc[index, 2]
            sample_item = {'I': I, 'mos': mos, 'std': std}
            sample.append(sample_item)

        return sample

    def __len__(self):
        return len(self.data[0].index)

