import numpy as np
from numpy.lib.type_check import imag
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

import torch
from tqdm import tqdm


def dataset_load(path, type='train'):
    if type == 'train':
        path = path + '/1.Training/원천데이터'
    elif type == 'val':
        path = path + '/2.Validation/원천데이터'

    car_dataset = []
    car_label = []
    car_label_num = []

    brand_list = os.listdir(path)
    # 맥환경에는 .DS_Store라는 파일이 들어감 제외해 준다.
    if '.DS_Store' in brand_list:
        brand_list.remove('.DS_Store')

    for i, brand in enumerate(brand_list):
        model_list = os.listdir(path+'/'+brand)

        if '.DS_Store' in model_list:
            model_list.remove('.DS_Store')

        for model in model_list:
            year_color_list = os.listdir(path+'/'+brand+'/'+model)

            if '.DS_Store' in year_color_list:
                year_color_list.remove('.DS_Store')

            for year_color in year_color_list:
                dataset = glob.glob(path+'/'+brand+'/' +
                                    model+'/'+year_color + '/*.jpg')
                if len(dataset) == 0:

                    continue
                car_dataset.append(dataset)
                car_label.append(brand+'_'+model+'_'+year_color)
                car_label_num.append(len(dataset))
                # if len(dataset) == 6:
                #     print(brand+'_'+model+'_'+year_color)
                #     print(car_dataset[-1])
                #     print(car_label[-1])
                #     print(car_label_num[-1])
                #     print(len(car_dataset), len(car_label), len(car_label_num))

    torch.save(car_label, './car_list')

    return (car_dataset, car_label_num)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = []
        self.class_list = []

        self.transform = None

        self.car_dataset, self.car_label_num = dataset_load(self.path, 'train')

        for i, img in enumerate(self.car_dataset):
            self.img_list += img
            self.class_list += [i] * self.car_label_num[i]

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

        self.car_dataset, self.car_label_num = dataset_load(self.path, 'val')

        for i, img in enumerate(self.car_dataset):
            self.img_list += img
            self.class_list += [i] * self.car_label_num[i]

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

def get_mean(dataset, type):
    path = './meanstd/'+type+'.mean'

    if os.path.isfile(path):
        print('Load %s dataset mean' % (type))
        return torch.load(path)

    print('Get %s dataset mean...' % (type))
    meanRGB = [np.mean(np.array(image), axis=(1, 2))
               for image, _ in tqdm(dataset, desc='meanRGB')]
    meanR = np.mean([m[0] for m in tqdm(meanRGB, desc='meanR')])
    meanG = np.mean([m[1] for m in tqdm(meanRGB, desc='meanG')])
    meanB = np.mean([m[2] for m in tqdm(meanRGB, desc='meanB')])

    torch.save([meanR, meanG, meanB], path)
    return [meanR, meanG, meanB]

# 채널 별 str 계산


def get_std(dataset, type):
    path = './meanstd/'+type+'.std'

    if os.path.isfile(path):
        print('Load %s dataset std' % (type))
        return torch.load(path)

    print('Get %s dataset std...' % (type))
    stdRGB = [np.std(np.array(image), axis=(1, 2))
              for image, _ in tqdm(dataset, desc='stdRGB')]
    stdR = np.mean([s[0] for s in tqdm(stdRGB, desc='stdR')])
    stdG = np.mean([s[1] for s in tqdm(stdRGB, desc='stdG')])
    stdB = np.mean([s[2] for s in tqdm(stdRGB, desc='stdB')])

    torch.save([stdR, stdG, stdB], path)
    return [stdR, stdG, stdB]
