import torch
import cv2
import pandas as pd
import numpy as np

data_file = pd.read_csv("C:\\Users\\shawn\\Documents\\TNI_Knees_RD_with_Gender\\TNI_Knees_RD_with_Gender\\knee_gender_labels.csv")
navigate = data_file['ID_IMG']
labels = data_file['SEX_LABEL']
encode_map = {
    'M': 1,
    'F': 0
}
labels.replace(encode_map, inplace=True)

folder_number = []
file_name = []

for i in navigate:
    folder_number.append(i.rpartition('/')[0])
    file_name.append(i.rpartition('/')[2])

overall_directory = "C:\\Users\\shawn\\Documents\\TNI_Knees_RD_with_Gender\\TNI_Knees_RD_with_Gender"
str_combo = []
img_doc = []
count = 0

for i in range(0, len(navigate)):
    str_combo.append(overall_directory + '\\' + folder_number[i] + '\\' + file_name[i])
    img_doc.append(cv2.imread(str_combo[i]))
    count += 1
    print(count)








