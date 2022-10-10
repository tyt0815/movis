import torch
from torch.utils.data import Dataset

import os
import glob
from PIL import Image
from os.path import isdir


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def dataset_load(path, type, transform=None):
    if type == 'train':
        path = path + '/1.Training/원천데이터'
    elif type == 'val':
        path = path + '/2.Validation/원천데이터'

    car_dataset = []
    car_label = []
    car_label_num = []
    img_list = []
    class_list = []

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
                year = (year_color.split('_'))[0]

                if isdir(path+'/'+brand+'/'+model+'/'+year_color) == False:
                    continue

                dataset = glob.glob(path+'/'+brand+'/' +
                                    model+'/'+year_color + '/*.jpg')

                if len(dataset) == 0:
                    continue

                if ((brand+'_'+model+'_'+year) in car_label) == False:
                    car_label.append(brand+'_'+model+'_'+year)

                img_list += dataset
                class_list += [car_label.index(brand +
                                               '_'+model+'_'+year)] * len(dataset)

    torch.save(car_label, './car_list')

    for i, data in enumerate(img_list):
        print(data)
        print(car_label[class_list[i]])

    dataset = DatasetFromFolder(img_list, class_list, transform)

    return dataset


class DatasetFromFolder(Dataset):
    def __init__(self, img_list, class_list, transform=None):

        self.transform = transform

        self.img_list = img_list
        self.class_list = class_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
