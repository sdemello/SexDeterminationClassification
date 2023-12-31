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

    #model = resnet18()
    model = vgg16_model()
    #model = EfficientNetModelImport()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    img_size = (224, 224)
    data_mean_overall = [0.1565, 0.1488, 0.1424]
    data_std_overall = [0.1438, 0.1412, 0.1361]
    # dataset_file = '/home/shawn/Documents/BinaryClassifier/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender/knee_gender_labels.csv'
    dataset_file = 'C:/Users/User/PycharmProjects/MachineLearning/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender/knee_gender_labels.csv'
    #weights_path = '/home/shawn/Documents/BinaryClassifier/Weights/'
    weights_path = 'C:/Users/User/PycharmProjects/MachineLearning/Weights/'
    #report_path = '/home/shawn/Documents/BinaryClassifier/Reports/'
    report_path = 'C:/Users/User/PycharmProjects/MachineLearning/Reports/'
    epochs = 20
    save_criteria = 'Accuracy'

    idsX_train, idsX_val, idsX_test, labelsY_train, labelsY_val, labelsY_test = dataset_split(dataset_file)
    idsX_train = idsX_train.to_numpy()
    idsX_val = idsX_val.to_numpy()
    idsX_test = idsX_test.to_numpy()
    labelsY_train = labelsY_train.to_numpy()
    labelsY_val = labelsY_val.to_numpy()
    labelsY_test = labelsY_test.to_numpy()

    train_loader = get_dataloader(idsX_train, labelsY_train, img_size, batch_size, data_mean_overall, data_std_overall, data_split = 'Training', shuffle=True)
    val_loader = get_dataloader(idsX_val, labelsY_val, img_size, batch_size, data_mean_overall, data_std_overall, data_split = 'Validation', shuffle=False)
    test_loader = get_dataloader(idsX_test, labelsY_test, img_size, batch_size, data_mean_overall, data_std_overall, data_split = 'Testing', shuffle=False)

    # Training and Validation
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    pathlib.Path(weights_path).mkdir(parents=True, exist_ok=True)
    train_validate(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_criteria, weights_path)

    # Testing
    weights_files_list = glob.glob(weights_path + '/*pth')
    weights_file = max(weights_files_list, key=os.path.getctime)
    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model Loaded!\nAccuracy: {:.4f}\nLoss: {:.4f}\nSensitivity: {:.4f}\nSpecificity: {:.4f}'.format(checkpoint['accuracy'], checkpoint['loss'], checkpoint['sensitivity'], checkpoint['specificity']))

    pathlib.Path(report_path).mkdir(parents=True, exist_ok=True)

    test_report(model, test_loader, criterion, device, report_path)


if __name__ == "__main__":
    main()




