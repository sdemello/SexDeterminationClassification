import random
import os
import pathlib
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from Models import resnet18
from DatasetFile import dataset_split, get_dataloader
from TrainFile import train_validate
from TestFile import test_report
from VGG16Model import vgg16_model
from keras.applications.vgg16 import VGG16
from EfficientNetModelFile import EfficientNetModelImport


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    img_size = (224, 224)
    # data_mean_overall = [0.1565, 0.1488, 0.1424]
    # data_std_overall = [0.1438, 0.1412, 0.1361]
    dataset_file = '/home/Projects/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender/knee_gender_labels.csv'
    save_criteria = 'Accuracy'
    plot_directory = '/home/Projects/'
    # classification_systems = ['RecessDistension', 'SexClassification']
    classification_systems = ['SexClassification']
    if classification_systems[0] == 'SexClassification':
        weights_paths = ['/home/Projects/WeightsSexClass/', '/home/Projects/WeightsRecessDist/']
        report_paths = ['/home/Projects/ReportsSexClass/', '/home/Projects/ReportsRecessDist/']
        saliency_img_directories = ['/home/Projects/SaliencyMapsSexClass/', '/home/Projects/SaliencyMapsRecessDist/']
        gradcam_img_directories = ['/home/Projects/GradCamMapsSexClass/']
    elif classification_systems[0] == 'RecessDistension':
        weights_paths = ['/home/Projects/WeightsRecessDist/', '/home/Projects/WeightsSexClass/']
        report_paths = ['/home/Projects/ReportsRecessDist/', '/home/Projects/ReportsSexClass/']
        saliency_img_directories = ['/home/Projects/SaliencyMapsRecessDist/', '/home/Projects/SaliencyMapsSexClass/']
    epochs = 30
    # epochs = 2
    dataset_limit = 1000

    for i in range(len(classification_systems)):
        print('Part: {}'.format(classification_systems[i]))
        # model = resnet18()
        # model = vgg16_model()
        model = EfficientNetModelImport()

        if i == 0:
            idsX_train, idsX_test, labelsY_train, labelsY_test, images_global, ids_global = dataset_split(dataset_file, classification_systems[i], i, dataset_limit)

            train_loader = get_dataloader(idsX_train, labelsY_train, img_size, batch_size, data_split='Training', shuffle=False)
            test_loader = get_dataloader(idsX_test, labelsY_test, img_size, batch_size, data_split='Testing', shuffle=False)

            images1_global = images_global
            ids1_global = ids_global

        else:
            idsX_train, idsX_test, labelsY_train, labelsY_test, images_global, ids_global = dataset_split(dataset_file, classification_systems[i], i, dataset_limit, images1_global, ids1_global)

            train_loader = get_dataloader(idsX_train, labelsY_train, img_size, batch_size, data_split='Training', shuffle=True)
            test_loader = get_dataloader(idsX_test, labelsY_test, img_size, batch_size, data_split='Testing', shuffle=False)

        # Training and Validation
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        pathlib.Path(weights_paths[i]).mkdir(parents=True, exist_ok=True)
        train_validate(model, train_loader, batch_size, optimizer, criterion, device, epochs, save_criteria, weights_paths[i], classification_systems[i], plot_directory)

        # Testing
        weights_files_list = glob.glob(weights_paths[i] + '/*pth')
        weights_file = max(weights_files_list, key=os.path.getctime)
	print(weights_file)
        checkpoint = torch.load(weights_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        print('Model Loaded!\nAccuracy: {:.4f}\nLoss: {:.4f}\nSensitivity: {:.4f}\nSpecificity: {:.4f}'.format(checkpoint['accuracy'], checkpoint['loss'], checkpoint['sensitivity'], checkpoint['specificity']))

        pathlib.Path(report_paths[i]).mkdir(parents=True, exist_ok=True)

        test_report(model, test_loader, criterion, device, report_paths[i], batch_size, classification_systems[i], gradcam_img_directories[i])


if __name__ == "__main__":
    main()




