import numpy as np
from numpy.lib.type_check import imag
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

import torch
from tqdm import tqdm

from os.path import isdir


def dataset_load(path, type='train'):
    if type == 'train':
        path = path + '/1.Training/원천데이터'
    elif type == 'val':
        path = path + '/2.Validation/원천데이터'

    car_dataset = []
    car_label = []
    car_label_num = []

    brand_list = os.listdir(path)

    for i, brand in enumerate(brand_list):
        if isdir(path+'/'+brand) == False:
            continue

        model_list = os.listdir(path+'/'+brand)

        for model in model_list:
            if isdir(path+'/'+brand+'/'+model) == False:
                continue
            year_color_list = os.listdir(path+'/'+brand+'/'+model)

            for year_color in year_color_list:

                if isdir(path+'/'+brand+'/'+model+'/'+year_color) == False:
                    continue

                dataset = glob.glob(path+'/'+brand+'/' +
                                    model+'/'+year_color + '/*.jpg')

                if len(dataset) == 0:
                    continue

                car_dataset.append(dataset)
                car_label.append(brand+'_'+model+'_'+year_color)
                car_label_num.append(len(dataset))

    torch.save(car_label, './car_list')

    img_list = []
    class_list = []

    for i, img in enumerate(car_dataset):
        img_list += img
        class_list += [i] * car_label_num[i]

    return (img_list, class_list)
    # return (car_dataset,car_label_num)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, path):
        self.path = path
        # self.img_list = []
        # self.class_list = []

        self.transform = None

        self.img_list, self.class_list = dataset_load(self.path, 'train')

        # self.car_dataset, self.car_label_num = dataset_load(self.path, 'train')

        # for i, img in enumerate(self.car_dataset):
        #     self.img_list += img
        #     self.class_list += [i] * self.car_label_num[i]

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
        # self.img_list = []
        # self.class_list = []

        self.transform = None

        self.img_list, self.class_list = dataset_load(self.path, 'val')

        # self.car_dataset, self.car_label_num = dataset_load(self.path, 'val')

        # for i, img in enumerate(self.car_dataset):
        #     self.img_list += img
        #     self.class_list += [i] * self.car_label_num[i]

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

    print('Get %s dataset mean...' % (type))
    meanRGB = [np.mean(np.array(image), axis=(1, 2))
               for image, _ in tqdm(dataset, desc='meanRGB')]
    meanR = np.mean([m[0] for m in tqdm(meanRGB, desc='meanR  ')])
    meanG = np.mean([m[1] for m in tqdm(meanRGB, desc='meanG  ')])
    meanB = np.mean([m[2] for m in tqdm(meanRGB, desc='meanB  ')])

    # torch.save([meanR, meanG, meanB], path)
    return [meanR, meanG, meanB]

# 채널 별 str 계산


def get_std(dataset):

    print('Get %s dataset std...' % (type))
    stdRGB = [np.std(np.array(image), axis=(1, 2))
              for image, _ in tqdm(dataset, desc='stdRGB')]
    stdR = np.mean([s[0] for s in tqdm(stdRGB, desc='stdR  ')])
    stdG = np.mean([s[1] for s in tqdm(stdRGB, desc='stdG  ')])
    stdB = np.mean([s[2] for s in tqdm(stdRGB, desc='stdB  ')])

    return [stdR, stdG, stdB]


def load_mean(dataset, type):
    path = './meanstd/'+type+'.mean'

    if os.path.isfile(path):
        print('Load %s dataset mean' % (type))
        return torch.load(path)

    meanR, meanG, meanB = get_mean(dataset)

    torch.save([meanR, meanG, meanB], path)

    return [meanR, meanG, meanB]


def load_std(dataset, type):
    path = './meanstd/'+type+'.std'

    if os.path.isfile(path):
        print('Load %s dataset std' % (type))
        return torch.load(path)

    stdR, stdG, stdB = get_std(dataset)

    torch.save([stdR, stdG, stdB], path)

    return [stdR, stdG, stdB]
