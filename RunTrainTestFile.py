import random
import os
import pathlib
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from Models import resnet18
from DatasetFile import get_dataloader
from TrainFile import train_validate
from DatasetFile import dataset_split
from TestFile import test_report

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    seed_everything(123)

    model = resnet18()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #train_file = []
    #val_file = []
    batch_size = 32
    img_size = (678, 975)
    data_mean_overall = [0.1565, 0.1488, 0.1424]
    data_std_overall = [0.1438, 0.1412, 0.1361]
    weights_path = './home/shawn/Documents/BinaryClassifier/Weights/'
    report_path = './home/shawn/Documents/BinaryClassifier/Reports/'
    epochs = 25
    save_criteria = 'Accuracy'
    dataset_file = '/home/shawn/Documents/BinaryClassifier/TNI_Knees_RD_with_Gender/TNI_Knees_RD_with_Gender/knee_gender_labels.csv'
    #test_file_path = './home/shawn/Documents/BinaryClassifier/TestFolder/'

    idsX_train, idsX_val, idsX_test, labelsY_train, labelsY_val, labelsY_test = dataset_split(dataset_file)

    train_loader = get_dataloader(idsX_train, labelsY_train, img_size, batch_size, data_mean_overall, data_std_overall)

    val_loader = get_dataloader(idsX_val, labelsY_val, img_size, batch_size, data_mean_overall, data_std_overall)

    test_loader = get_dataloader(idsX_test, labelsY_test, img_size, batch_size, data_mean_overall, data_std_overall)

    # Training and Validation
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    pathlib.Path(weights_path).mkdir(parents=True, exist_ok=True)

    train_validate(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_criteria, weights_path)

    # Testing
    weights_file = glob.glob(weights_path + '/*pth')[0]
    checkpoint = torch.load(weights_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model Loaded!\nAccuracy: {:.4}\nLoss: {:.4}\nSensitivity: {:.4}\nSpecificity: {:.4}'.format(checkpoint['accuracy'], checkpoint['loss'], checkpoint['sensitivity'], checkpoint['specificity']))

    pathlib.Path(report_path).mkdir(parents=True, exist_ok=True)

    test_report(model, test_loader, criterion, device, report_path)


if __name__ == "__main__":
    main()




