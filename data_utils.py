# import torch
# from torchvision import transforms

import numpy as np
from numpy.lib.type_check import imag
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os


def dataset_load(path, type='train'):
    path = path+'/원천데이터'
    car_dataset = []
    car_label = []

    brand_list = os.listdir(path)
    # colab 환경용. colab이 아닐시 주석처리
    if '.DS_Store' in brand_list:
        brand_list.remove('.DS_Store')

    for i, brand in enumerate(brand_list):
        # print(i, brand)
        model_list = os.listdir(path+'/'+brand)
        # colab 환경용. colab이 아닐시 주석처리
        if '.DS_Store' in model_list:
            model_list.remove('.DS_Store')
        # print(model_list)
        for model in model_list:
            year_color_list = os.listdir(path+'/'+brand+'/'+model)
            # for year_color in year_color_list:
            #     dataset = glob.glob(path+'/'+brand+'/'+model+'/'+year_color + '/*.jpg')
            #     car_dataset.append(dataset)
            #     car_label.append(brand+'_'+model+'_'+year_color)
            dataset = []
            for year_color in year_color_list:
                dataset += glob.glob(path+'/'+brand+'/' +
                                     model+'/'+year_color + '/*.jpg')
            car_dataset.append(dataset)
            car_label.append(brand+'_'+model)

    return (car_dataset, car_label)

# car_dataset , car_label = dataset_load('./drive/MyDrive/car_sample')
# print(car_label)
# print(car_dataset)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = []
        self.class_list = []

        self.transform = None

        # self.kia_path = path + '/kia/train'
        # self.bmw_path = path + '/bmw/train'

        # self.kia_img_list = glob.glob(self.kia_path + '/*.jpg')
        # self.bmw_img_list = glob.glob(self.bmw_path + '/*.jpg')

        # self.img_list = self.kia_img_list + self.bmw_img_list
        # self.class_list = [0] * len (self.kia_img_list)+ [1]*len(self.bmw_img_list)

        self.car_dataset, self.car_label = dataset_load(self.path, 'train')

        for i, img in enumerate(self.car_dataset):
            self.img_list += img
            self.class_list += [i] * len(img)
            # for j in range(len(img)):
            #     self.class_list.append(self.car_label[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ValDatasetFromFolder(Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = []
        self.class_list = []

        self.transform = None

        # self.kia_path = path + '/kia/test'
        # self.bmw_path = path + '/bmw/test'

        # self.kia_img_list = glob.glob(self.kia_path + '/*.jpg')
        # self.bmw_img_list = glob.glob(self.bmw_path + '/*.jpg')

        # self.img_list = self.kia_img_list + self.bmw_img_list
        # self.class_list = [0] * len (self.kia_img_list)+ [1]*len(self.bmw_img_list)

        self.car_dataset, self.car_label = dataset_load(self.path, 'train')

        for i, img in enumerate(self.car_dataset):
            self.img_list += img
            self.class_list += [i] * len(img)
            # for j in range(len(img)):
            #     self.class_list.append(self.car_label[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


# 채널 별 mean 계산

def get_mean(dataset):
    meanRGB = [np.mean(np.array(image), axis=(1, 2)) for image, _ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return [meanR, meanG, meanB]

# 채널 별 str 계산


def get_std(dataset):
    stdRGB = [np.std(np.array(image), axis=(1, 2)) for image, _ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [stdR, stdG, stdB]
