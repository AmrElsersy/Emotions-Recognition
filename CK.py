import argparse
import cv2
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import ToTensor, ToPILImage, RandomCrop, RandomCrop
from torchvision.transforms.transforms import RandomRotation, RandomHorizontalFlip, Compose 
import pandas as pd
import os
import numpy as np
from utils import normalization

def random_rotation(image):
    h, w = image.shape[0:2]
    center = (w//2, h//2)
    angle =  int(np.random.randint(-10, 10))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, image.shape)
    return image

def numpy_loader(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = random_rotation(image)
    image = normalization(image)
    return image

def create_CK_dataloader(root='data/CK+48', batch_size=1):
    # transform = Compose([ToPILImage(), RandomRotation(45), RandomHorizontalFlip(0.5), ToTensor()])
    transform = Compose([ToPILImage(), RandomCrop(45), RandomHorizontalFlip(0.5), ToTensor()])
    image_folder = ImageFolder(root, loader=numpy_loader, transform=transform)
    # transform = Compose([RandomRotation(20), RandomHorizontalFlip(0.5), ToTensor()])
    # image_folder = ImageFolder(root, transform=transform)

    # ====== Split =========
    indices = list(range(len(image_folder)))
    train_indices = indices[:700]
    test_indices = indices[700:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    dataloader = DataLoader(image_folder,  batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(image_folder, batch_size=batch_size, sampler=test_sampler)
    return dataloader, test_dataloader

if __name__ == '__main__':
    dataloader, test_dataloader =create_CK_dataloader()
    print(len(dataloader))

    for image, label in dataloader:
        image = image.squeeze().numpy()
        print(image.shape)
        print(image)
        print(label,'\n')    
        # cv2.imshow('image', image[0])
        cv2.imshow('image', image)
        if cv2.waitKey(0) == 27:
            break

