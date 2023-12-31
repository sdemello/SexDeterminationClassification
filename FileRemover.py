import torch
import pandas as pd
import numpy as np
import glob
import os

file_path = 'C:\\Users\\User\\PycharmProjects\\MachineLearning\\TNI_Knees_RD_with_Gender - Copy\\TNI_Knees_RD_with_Gender\\'
data1 = pd.read_csv("C:\\Users\\User\\PycharmProjects\\MachineLearning\\TNI_Knees_RD_with_Gender - Copy\\TNI_Knees_RD_with_Gender\\knee_gender_labels.csv")
img_directories = data1['ID_IMG']
img_directories_str = []
img_folder = []
img_file = []
for i in img_directories:
    img_folder.append(i.rpartition('/')[0])
    img_file.append(i.rpartition('/')[2])
print(len(img_folder))
print(len(img_file))
recess_dst1 = data1['RD_LABEL']
print(len(data1))

'''
for i in range(len(data1)):
    if recess_dst1[i] == 1.0:
        file_directory = file_path + '\\' + img_folder[i] + '\\' + img_file[i]
        os.remove(file_directory)
        data1.drop([i], axis=0, inplace=True)
'''

#print(len(data1))

print(img_folder[0:5])
dir = []
for i in range(len(data1)):
    folder_directory = file_path + img_folder[i]
    if os.path.isdir(folder_directory):
        dir_folder = os.listdir(folder_directory)
        if len(dir_folder) == 0:
            os.rmdir(folder_directory)
            print("Deleted {}".format(img_folder[i]))
        else:
            continue
    else:
        continue




