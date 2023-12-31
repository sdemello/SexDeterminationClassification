import numpy as np
import pandas as pd
import os

#data1 = pd.read_csv("C:\\Users\\User\\PycharmProjects\\MachineLearning\\TNI_Knees_RD_with_Gender - Copy\\TNI_Knees_RD_with_Gender\\knee_gender_labels.csv")
data1 = pd.read_csv("/home/Projects/TNI_Knees_RD_with_Gender - Smaller Copy/TNI_Knees_RD_with_Gender/knee_gender_labels.csv")
directorypath = data1['ID_IMG']
img_folder = []
img_file = []
for i in directorypath:
    img_folder.append(i.rpartition('/')[0])
    img_file.append(i.rpartition('/')[2])
#folderpath = directorypath.rpartition('/')[0]
#filepath = directorypath.rpartition('/')[2]
sex_labels = data1['SEX_LABEL']
recess_labels = data1['RD_LABEL']

data2 = pd.DataFrame(columns=['ID_IMG', 'SEX_LABEL', 'RD_LABEL'])
#base_directory = "C:/Users/User/PycharmProjects/MachineLearning/TNI_Knees_RD_with_Gender - Copy/TNI_Knees_RD_with_Gender/"
base_directory = '/home/Projects/TNI_Knees_RD_with_Gender - Smaller Copy/TNI_Knees_RD_with_Gender/'
temp_directory_file = ""

for i in range(len(data1)):
    temp_directory_file = base_directory + '/' + img_folder[i] + '/' + img_file[i]
    if os.path.isfile(temp_directory_file):
        #data2.append(data1[i])
        data2.loc[i] = [directorypath[i], sex_labels[i], recess_labels[i]]

data2 = data2.reset_index(drop=True)
report_path = '/home/Projects/TNI_Knees_RD_with_Gender - Smaller Copy/TNI_Knees_RD_with_Gender/'
save_file_name = 'knee_gender_labels2_0'
data2.to_csv(report_path + save_file_name + '.csv')










