# CropImage.py
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings
warnings.filterwarnings('ignore')


def CropImage(orig_img, index):
    orig_img_rows, orig_img_columns, channels = orig_img.shape
    altered_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

    gX = cv2.Sobel(altered_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    gY = cv2.Sobel(altered_img, ddepth=cv2.CV_32F, dx=0, dy=1)

    gX_img = cv2.convertScaleAbs(gX)
    gY_img = cv2.convertScaleAbs(gY)

    combined_sobel = cv2.addWeighted(gX_img, 0.5, gY_img, 0.5, 0)

    def getVerticalImageByDY(img):
        img_rows, img_columns = img.shape
        revised_vertical_img = np.zeros((img_rows, img_columns-1), np.uint8)

        for i in range(img_rows):
            for j in range(img_columns-1):
                revised_vertical_img[i, j] = abs(img[i, j+1] - img[i, j])

        return revised_vertical_img

    def getHorizontalImageByDX(img):
        img_rows, img_columns = img.shape
        revised_horizontal_img = np.empty((img_rows-1, img_columns))

        for j in range(img_columns):
            for i in range(img_rows-1):
                revised_horizontal_img[i, j] = abs(img[i+1, j] - img[i, j])

        return revised_horizontal_img

    def getSignificantYIndexes(img, threshold, percent_consideration):
        img_rows, img_columns = img.shape
        significant_y_indexes = []

        for i in range(img_rows-1):
            dY_temp_array = []
            count_some_index = 0
            for j in range(img_columns):
                dY = abs(img[i+1, j] - img[i, j])
                dY_temp_array.append(dY)
            for k in range(len(dY_temp_array)):
                if dY_temp_array[k] > threshold:
                    count_some_index += 1
            if (count_some_index > (percent_consideration*img_columns)) and (count_some_index < (0.8*img_columns)):
                significant_y_indexes.append(i)

        return significant_y_indexes

    def getSignificantXIndexes(img, threshold, percent_consideration):
        img_rows, img_columns = img.shape
        significant_x_indexes = []

        for j in range(img_columns-1):
            dX_temp_array = []
            count_some_index = 0
            for i in range(img_rows):
                dX = abs(img[i, j+1] - img[i, j])
                dX_temp_array.append(dX)
            for k in range(len(dX_temp_array)):
                if dX_temp_array[k] > threshold:
                    count_some_index += 1
            if (count_some_index > (percent_consideration*img_rows)) and (count_some_index < (0.8*img_rows)):
                significant_x_indexes.append(j)

        return significant_x_indexes

    def getBreakoutXIndexes(vertical_image, y_index, threshold):
        img_rows, img_columns = vertical_image.shape
        important_x_indexes = []

        for j in range(img_columns-1):
            dX = abs(vertical_image[y_index+2, j+1] - vertical_image[y_index+2, j])
            if dX > threshold:
                important_x_indexes.append(j)

        refined_first_important_x_indexes = []
        for i in range(len(important_x_indexes)):
            if important_x_indexes[i]-6 > 0:
                if (vertical_image[y_index+2, important_x_indexes[i]-3] == 0) and (
                        vertical_image[y_index+2, important_x_indexes[i]-4] == 0) and (
                        vertical_image[y_index+2, important_x_indexes[i]-5] == 0) and (
                        vertical_image[y_index+2, important_x_indexes[i]-6] == 0) and (
                        important_x_indexes[i] > 0.05*img_columns):
                    refined_first_important_x_indexes.append(important_x_indexes[i])

        refined_second_important_x_indexes = []
        for i in range(len(important_x_indexes)):
            if important_x_indexes[i]+6 < img_columns-1:
                if (vertical_image[y_index+2, important_x_indexes[i]+3] == 0) and (
                        vertical_image[y_index+2, important_x_indexes[i]+4] == 0) and (
                        vertical_image[y_index+2, important_x_indexes[i]+5] == 0) and (
                        vertical_image[y_index+2, important_x_indexes[i]+6] == 0):
                    refined_second_important_x_indexes.append(important_x_indexes[i])

        if (len(refined_first_important_x_indexes) != 0) and (len(refined_second_important_x_indexes) != 0):
            breakout_x_indexes = [refined_first_important_x_indexes[0], refined_second_important_x_indexes[-1]]

        else:
            breakout_x_indexes = [1, img_columns-1]

        return breakout_x_indexes

    combined_sobel_revised_vertical = getVerticalImageByDY(combined_sobel)
    combined_sobel_revised_horizontal = getHorizontalImageByDX(combined_sobel)

    threshold1 = 50
    horizontal_percent_consideration = 0.5
    y_indexes = getSignificantYIndexes(combined_sobel_revised_horizontal, threshold1, horizontal_percent_consideration)
    if len(y_indexes) == 0:
        horizontal_percent_consideration = 0.25
        y_indexes = getSignificantYIndexes(combined_sobel_revised_horizontal, threshold1, horizontal_percent_consideration)
    if len(y_indexes) != 0:
        two_most_important_y_indexes = [y_indexes[0], y_indexes[-1]]

    threshold2 = 100
    vertical_percent_consideration = 0.5
    x_indexes = getSignificantXIndexes(combined_sobel_revised_vertical, threshold2, vertical_percent_consideration)
    if len(x_indexes) == 0:
        # vertical_percent_consideration = 0.3
        # x_indexes = getSignificantXIndexes(combined_sobel_revised_vertical, threshold2, vertical_percent_consideration)
        x_indexes = getBreakoutXIndexes(combined_sobel_revised_vertical, y_indexes[0], threshold2)
    if len(x_indexes) != 0:
        two_most_important_x_indexes = [x_indexes[0], x_indexes[-1]]

    if two_most_important_x_indexes[0] > 0.5*orig_img_columns:
        two_most_important_x_indexes[0] = 0

    if two_most_important_y_indexes[0] > 0.5*orig_img_rows:
        two_most_important_y_indexes[0] = 0

    if two_most_important_x_indexes[1] < 0.6*orig_img_columns:
        two_most_important_x_indexes[1] = orig_img_columns-1

    if two_most_important_y_indexes[1] < 0.65*orig_img_rows:
        two_most_important_y_indexes[1] = orig_img_rows-1

    orig_img_revised_crop = orig_img[two_most_important_y_indexes[0]:two_most_important_y_indexes[1], two_most_important_x_indexes[0]:two_most_important_x_indexes[1]]
    # if index % 50 == 0:
    #     print(f'Image {index+1} Cropped')

    return orig_img_revised_crop













