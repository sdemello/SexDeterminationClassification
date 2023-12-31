import pandas as pd
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa


class CustomDataset(Dataset):
    def __init__(self, ids, labels, transf):
        super().__init__()
        self.transforms = transf
        self.ids = ids
        # self.labels = torch.LongTensor(labels.values)
        self.labels = torch.LongTensor(labels)
        self.data_len = len(self.ids)

    def __getitem__(self, index):
        id_img = self.ids[index]

        folder_number = id_img.rpartition('/')[0]
        file_name = id_img.rpartition('/')[2]
        # overall_directory = '/home/shawn/Documents/BinaryClassifier/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender'
        overall_directory = 'C:/Users/User/PycharmProjects/MachineLearning/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender'
        file_directory = (overall_directory + '/' + folder_number + '/' + file_name)

        img = cv2.imread(file_directory)
        img = cv2.resize(img, (224, 224))
        img = cv2.convertScaleAbs(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transforms:
            img = self.transforms(img)

        label = self.labels[index]

        return id_img, img, label

    def __len__(self):
        return self.data_len


class ImgAugTransform(object):
    def __init__(self):
        self.aug = iaa.Sequential([
            # Blur or Sharpness
            iaa.Sometimes(0.25, iaa.OneOf([iaa.GaussianBlur(sigma=(0, 1.0)), iaa.pillike.EnhanceSharpness(factor=(0.8, 1.5))])),
            # Flip horizontally
            iaa.Fliplr(0.5),
            # Rotation
            iaa.Rotate((-20, 20)),
            # Pixel Dropout
            iaa.Sometimes(0.25, iaa.OneOf([iaa.Dropout(p=(0, 0.1)), iaa.CoarseDropout(0.1, size_percent=0.5)])),
            # Color
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def dataset_split(dir_img):
    images = pd.read_csv(dir_img)
    recess_dst = images['RD_LABEL']

    print(len(images))
    for i in range(0, len(recess_dst)):
        if recess_dst[i] == 1.0:
            images.drop([i], axis=0, inplace=True)

    print(len(images))

    images1 = images.sample(frac=0.05)
    ids = images1['ID_IMG']
    labels = images1['SEX_LABEL']
    encode_map = {
        'M': 1,
        'F': 0
    }
    labels.replace(encode_map, inplace=True)
    print("Male Count: {}".format(labels.value_counts()[1]))
    print("Female Count: {}".format(labels.value_counts()[0]))
    idsX_train, idsX_test, labelsY_train, labelsY_test = train_test_split(ids, labels, test_size=0.2, random_state=None)
    idsX_train, idsX_val, labelsY_train, labelsY_val = train_test_split(idsX_train, labelsY_train, test_size=0.25, random_state=None)
    return idsX_train, idsX_val, idsX_test, labelsY_train, labelsY_val, labelsY_test


def get_dataloader(ids, labels, img_size, batch_size, data_mean, data_std, data_split, shuffle):
    # ids, labels = read_dataset(data_file)
    double_img_size = tuple([2*x for x in img_size])

    if data_split == 'Training':
        test_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(double_img_size, Image.BICUBIC),

            # Augmentation
            ImgAugTransform(),

            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])

    if data_split == 'Validation' or data_split == 'Testing':
        test_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(double_img_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std)
        ])

    print(data_split + " Dataset Size: ", len(ids))

    dataset = CustomDataset(ids=ids, labels=labels, transf=test_transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader





