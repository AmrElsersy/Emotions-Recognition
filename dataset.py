"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: FER2013 dataset
"""
import argparse
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.transforms as transforms
import pandas as pd
import os
import numpy as np
import torch

from utils import get_label_emotion, normalization, histogram_equalization, standerlization, normalize_dataset_mode_1, normalize_dataset_mode_255, get_transforms

class FER2013(Dataset):
    """
    FER2013 format:
        index   emotion     pixels      Usage

    index: id of series
    emotion: label (from 0 - 6)
    pixels: 48x48 pixel value (uint8)
    Usage: [Training, PrivateTest, PublicTest]    
    """
    def __init__(self, root='../data', mode = 'train', transform = None):
        
        self.root = root
        self.transform = transform
        assert mode in ['train', 'val', 'test']
        self.mode = mode

        self.csv_path = os.path.join(self.root, 'fer2013.csv')
        self.df = pd.read_csv(self.csv_path)
        # print(self.df)

        if self.mode == 'train':
            self.df = self.df[self.df['Usage'] == 'Training']
        elif self.mode == 'val':
            self.df = self.df[self.df['Usage'] == 'PrivateTest']
        else:
            self.df = self.df[self.df['Usage'] == 'PublicTest']

    def __getitem__(self, index: int):
        data_series = self.df.iloc[index]
        emotion = data_series['emotion']
        pixels  = data_series['pixels']

        # to numpy
        face = list(map(int, pixels.split(' ')))
        face = np.array(face).reshape(48,48).astype(np.uint8)

        if self.transform:
            face = histogram_equalization(face)
            # face = normalization(face)
            face = self.transform(face)

        return face, emotion

    def __len__(self) -> int:
        return self.df.index.size


def create_train_dataloader(root='../data', batch_size=64):
    dataset = FER2013(root, mode='train', transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    return dataloader

def create_val_dataloader(root='../data', batch_size=2):
    dataset = FER2013(root, mode='val', transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader

def create_test_dataloader(root='../data', batch_size=1):
    # transform = transforms.ToTensor()
    transform = get_transforms()
    dataset = FER2013(root, mode='test', transform=transform)
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    return dataloader

def calculate_dataset_mean_std(dataset:FER2013):
    n = len(dataset)
    means = []
    stds = []
    for i in range(n):
        image, _ = dataset[i]
        # image = image/ 255
        mean = np.mean(image)
        std = np.std(image)
        means.append(mean)
        stds.append(std)
        print(f'i={i}, mean = {mean}, std = {std}')
    mean = np.mean(means)
    std = np.mean(stds)
    print(f'\n\t Mean = {mean} ... Std = {std}\n')

def test_dataloader_main():
    dataloader = create_test_dataloader()    
    for image, label in dataloader:
        image = image.squeeze().numpy()
        cv2.imshow('img', image)
        print(image.shape)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='train', help='dataset mode')    
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--mean_std', action='store_true', help='calculate mean std of dataset')
    parser.add_argument('--test', action='store_true', help='test augumentation')
    
    args = parser.parse_args()

    if args.test:
        test_dataloader_main()
        exit(0)

    dataset = FER2013(args.datapath, args.mode)
    print(f'dataset size = {len(dataset)}')
    
    if args.mean_std:
        calculate_dataset_mean_std(dataset)
        exit(0)
 
    for i in range(len(dataset)):

        face, emotion = dataset[i]
        # print('emotion',emotion)
        # print('shape',face.shape)
        face = np.copy(face)
        print(f"before min:{np.min(face)}, max:{np.max(face)}, mean:{np.mean(face)}, std:{np.std(face)}")
        face1 = normalize_dataset_mode_255(face)
        # face1 = normalization(face)
        face2 = standerlization(face)
        print(f"after min:{np.min(face)}, max:{np.max(face)}, mean:{np.mean(face)}, std:{np.std(face)}\n")

        face = cv2.resize(face, (200,200))
        cv2.putText(face, get_label_emotion(emotion), (0,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
        cv2.imshow('face', face)

        face1 = cv2.resize(face1, (200,200))
        face2 = cv2.resize(face2, (200,200))
        cv2.imshow('normalization', face1)
        cv2.imshow('standerlization', face2)

        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break






    # df = pd.DataFrame({
    #     "name": ["amr",'ELSERSY', 'sersy'],
    #     'salary': [100,20,3000],
    #     'job': ['software', 'ray2', 'mech']
    # }, index=[0,1,4])

    # print(df.index.size)

    # print(df)

    # ser = df[df['salary'] > 50]
    # print(type(ser)) # dataframe
    # print(ser)

    # salary = df[df['salary'] < 200]  # dataframe
    # print(type(salary.iloc[1])) # series
    # print(salary.iloc[1],'\n')
    # print(salary.iloc[0]['name'])
    # print(salary.shape)
    # print(salary.iloc[0]['name'])

    # df = df[df['salary'] == 3000] 
    # print(df[['name', "salary"]])

    # df = df[df['salary'] < 500]
    # print(df.index)
    # print(df.iloc[0].values)
    # print(type(df.values))
    # print('==========================')
    # # print(df.count()) # count of each series in the dataframe
    # print(df.iloc[0].count())
