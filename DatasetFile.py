import os

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from CropImage import CropImage


class CustomDataset(Dataset):
    def __init__(self, ids, imgs, labels, transf, phase):
        super().__init__()
        self.transforms = transf
        self.ids = ids
	self.imgs = imgs
        # self.labels = torch.LongTensor(labels.values)
        self.labels = torch.LongTensor(labels)
        self.phase = phase
        self.data_len = len(self.ids)

    def __getitem__(self, index):
        id_img = self.ids[index]

	img = self.imgs[index]
	
        img = cv2.resize(img, (224, 224))
        img = cv2.convertScaleAbs(img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transforms:
            img = self.transforms(img)

        # Find data_mean_img and data_std_img
        data_mean_img, data_std_img = find_mean_and_std(img)

        transforms2 = transforms.Compose([transforms.Normalize(data_mean_img, data_std_img)])
        img = transforms2(img)

        label = self.labels[index]

        if self.phase == 'Testing':
            data_mean_img = np.asarray(data_mean_img, dtype=np.float32)
            data_std_img = np.asarray(data_std_img, dtype=np.float32)
            return id_img, img, label, data_mean_img, data_std_img
        else:
            return id_img, img, label

    def __len__(self):
        return self.data_len


class ImgAugTransform(object):
    def __init__(self):
        self.aug = iaa.Sequential([
            # Blur or Sharpness
            iaa.Sometimes(0.25, iaa.OneOf(
                [iaa.GaussianBlur(sigma=(0, 1.0)), iaa.pillike.EnhanceSharpness(factor=(0.8, 1.5))])),
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


def dataset_split(dir_img, classification_system, index_number, dataset_limit, images1_global=None, ids_global=None):
    images = pd.read_csv(dir_img)
    recess_dst = images['RD_LABEL']
    sex_label = images['SEX_LABEL']
    male_and_recess_distension_count = 0
    male_and_no_recess_distension_count = 0
    female_and_recess_distension_count = 0
    female_and_no_recess_distension_count = 0
    print(len(images))
    male_images = images[sex_label == 'M']
    female_images = images[sex_label == 'F']
    RANDOM_STATE = 22

    '''
    # Drop Recess Distension
    for i in range(0, len(recess_dst)):
        if recess_dst[i] == 1.0:
            images.drop([i], axis=0, inplace=True)
    print(len(images))

    # Drop Male Only
    for i in range(len(sex_label)):
        if sex_label[i] == 'M':
            images.drop([i], axis=0, inplace=True)
    print(len(images))

    # Drop Female Only
    for i in range(len(sex_label)):
        if sex_label[i] == 'F':
            images.drop([i], axis=0, inplace=True)
    print(len(images))
    '''

    if index_number == 0:
        images_male = images[images['SEX_LABEL'] == 'M']
        images_female = images[images['SEX_LABEL'] == 'F']
        new_dataset_limit = dataset_limit // 2
        images1_male = choose_img_by_patient(images_male, new_dataset_limit)
        images1_female = choose_img_by_patient(images_female, new_dataset_limit)
        images1 = pd.concat([images1_male, images1_female]).reset_index(drop=True)
        images1 = images1.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        # print(images1)

        ids = images1['ID_IMG']
        images1_global = images1
        ids_global = ids
        print(f"Revised Dataset Size: {len(images1)}")

    if classification_system == 'SexClassification':
        if index_number != 0:
            ids = ids_global
            images1 = images1_global
        labels = images1['SEX_LABEL']
        encode_map = {
            'M': 1,
            'F': 0
        }
        labels.replace(encode_map, inplace=True)
        print("Male Count: {}".format(labels.value_counts()[1]))
        print("Female Count: {}".format(labels.value_counts()[0]))

    elif classification_system == 'RecessDistension':
        if index_number != 0:
            ids = ids_global
            images1 = images1_global
        labels = images1['RD_LABEL'].astype(int)
        sex_col = images1['SEX_LABEL']
        encode_map = {
            'M': 1,
            'F': 0
        }
        sex_col.replace(encode_map, inplace=True)
        print("Male Count: {}".format(sex_col.value_counts()[1]))
        print("Female Count: {}".format(sex_col.value_counts()[0]))
        print("Recess Dist Count: {}".format(labels.value_counts()[1]))

        for i in range(len(images1_global)):
            if sex_col[i] == 1 and labels[i] == 1:
                male_and_recess_distension_count += 1
            elif sex_col[i] == 1 and labels[i] == 0:
                male_and_no_recess_distension_count += 1
            elif sex_col[i] == 0 and labels[i] == 1:
                female_and_recess_distension_count += 1
            elif sex_col[i] == 0 and labels[i] == 0:
                female_and_no_recess_distension_count += 1

        print("Male Count (with Recess Distension): ", male_and_recess_distension_count)
        print("Male Count (without Recess Distension): ", male_and_no_recess_distension_count)
        print("Female Count (with Recess Distension): ", female_and_recess_distension_count)
        print("Female Count (without Recess Distension): ", female_and_no_recess_distension_count)

    images1_male_train = images1_male.sample(frac=0.8, random_state=RANDOM_STATE)
    images1_female_train = images1_female.sample(frac=0.8, random_state=RANDOM_STATE)

    images1_male_test = images1_male
    folder_numbers_male_train = images1_male_train['ID_IMG'].tolist()
    folder_numbers_male_train = [i.rpartition('/')[0] for i in folder_numbers_male_train]
    folder_numbers_train_male_overall = images1_male['ID_IMG'].tolist()
    folder_numbers_train_male_overall = [i.rpartition('/')[0] for i in folder_numbers_train_male_overall]
    for i in range(len(folder_numbers_male_train)):
        if folder_numbers_male_train[i] in folder_numbers_train_male_overall:
            images1_male_test.drop([i], axis=0, inplace=True)
    images1_male_test.reset_index(drop=True)

    images1_female_test = images1_female
    folder_numbers_female_train = images1_female_train['ID_IMG'].tolist()
    folder_numbers_female_train = [i.rpartition('/')[0] for i in folder_numbers_female_train]
    folder_numbers_train_female_overall = images1_female['ID_IMG'].tolist()
    folder_numbers_train_female_overall = [i.rpartition('/')[0] for i in folder_numbers_train_female_overall]
    for i in range(len(folder_numbers_female_train)):
        if folder_numbers_female_train[i] in folder_numbers_train_female_overall:
            images1_female_test.drop([i], axis=0, inplace=True)
    images1_female_test.reset_index(drop=True)

    # images1_male_test = images1_male.sample(frac=0.2, random_state=RANDOM_STATE)
    # images1_female_test = images1_female.sample(frac=0.2, random_state=RANDOM_STATE)

    images1_train = pd.concat([images1_male_train, images1_female_train]).reset_index(drop=True)
    images1_train = images1_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    images1_test = pd.concat([images1_male_test, images1_female_test]).reset_index(drop=True)
    images1_test = images1_test.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    idsX_train = images1_train['ID_IMG']
    idsX_test = images1_test['ID_IMG']

    if classification_system == 'SexClassification':
        labelsY_train = images1_train['SEX_LABEL']
        labelsY_test = images1_test['SEX_LABEL']

        encode_map = {
            'M': 1,
            'F': 0
        }

        labelsY_train.replace(encode_map, inplace=True)
        labelsY_test.replace(encode_map, inplace=True)

    elif classification_system == 'RecessDistension':
        labelsY_train = images1_train['RD_LABEL']
        labelsY_test = images1_test['RD_LABEL']

    idsX_train = idsX_train.to_numpy(); idsX_test = idsX_test.to_numpy()
    labelsY_train = labelsY_train.to_numpy(); labelsY_test = labelsY_test.to_numpy()

    return idsX_train, idsX_test, labelsY_train, labelsY_test, images1_global, ids_global


def get_dataloader(ids, labels, img_size, batch_size, data_split, shuffle):
    double_img_size = tuple([2*x for x in img_size])

    if data_split == 'Training':
        test_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(double_img_size, Image.BICUBIC),

            ImgAugTransform(),  # Augmentation

            transforms.ToTensor(),
            # transforms.Normalize(data_mean, data_std)
        ])

    if data_split == 'Validation' or data_split == 'Testing':
        test_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(double_img_size, Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize(data_mean, data_std)
        ])

    print(data_split + " Dataset Size: ", len(ids))

    overall_directory = 'C:/Users/User/PycharmProjects/MachineLearning/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender/'
    img_dataset = []
    for i in range(len(ids)):
        img = cv2.imread(overall_directory + ids[i])
        # img = CropImage(img, i)
        img_dataset.append(img)

    dataset = CustomDataset(ids=ids, imgs=img_dataset, labels=labels, transf=test_transform, phase=data_split)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def find_mean_and_std(img):
    R_mean, G_mean, B_mean = torch.mean(img, dim=[1, 2])
    data_mean = [R_mean, G_mean, B_mean]

    R_std, G_std, B_std = torch.std(img, dim=[1, 2])
    data_std = [R_std, G_std, B_std]

    return data_mean, data_std


def choose_random_image(folder_number):
    directory = '/home/Projects/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender/'
    directory2 = directory + folder_number
    dir_list = os.listdir(directory2)
    allImages = []
    for img in range(len(dir_list)):
        allImages.append(img)
    # allImages = [img for img in dir_list]
    choice = random.randint(0, len(allImages) - 1)
    chosenImage = dir_list[choice]
    chosenImage_revised = str(folder_number) + '/' + chosenImage

    # Get Corresponding Labels
    csv_file = directory + 'knee_gender_labels.csv'
    images_df = pd.read_csv(csv_file)
    id_imgs = images_df['ID_IMG']
    sex_label = images_df['SEX_LABEL']
    recess_dist_label = images_df['RD_LABEL']
    for i in range(len(id_imgs)):
        if id_imgs[i] == chosenImage_revised:
            corresponding_sex_label = sex_label[i]
            corresponding_rd_label = recess_dist_label[i]

    return chosenImage_revised, corresponding_sex_label, corresponding_rd_label


def choose_img_by_patient(dataframe, dataset_limit):
    id_img = dataframe['ID_IMG'].tolist()
    sex_label = dataframe['SEX_LABEL'].tolist()

    # Get a list of all the folders
    all_folder_numbers = [i.rpartition('/')[0] for i in id_img]

    # Choose random folders from list of all folders
    folder_numbers_random = []
    while len(folder_numbers_random) < dataset_limit:
        temp_folder_choice = random.choice(all_folder_numbers)
        if temp_folder_choice not in folder_numbers_random:
            folder_numbers_random.append(temp_folder_choice)

    # Get new dataframe, which consists of images chosen by patient
    new_dataframe = pd.DataFrame(columns=['ID_IMG', 'SEX_LABEL', 'RD_LABEL'])
    for i in range(len(folder_numbers_random)):
        chosenImage, sex_label, rd_label = choose_random_image(folder_numbers_random[i])
        new_dataframe.loc[i] = [chosenImage, sex_label, rd_label]

    return new_dataframe
















