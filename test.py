"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Training & Validation
"""
import numpy as np 
import argparse
import logging
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.tensorboard as tensorboard
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True
import cv2
import torchvision.transforms.transforms as transforms

from model.model import Mini_Xception
from model.depthwise_conv import SeparableConv2D
from Utils.dataset import create_train_dataloader, create_val_dataloader, FER2013
from Utils.utils import get_label_emotion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help="training batch size")
    parser.add_argument('--tensorboard', type=str, default='checkpoint/tensorboard', help='path log dir of tensorboard')
    parser.add_argument('--logging', type=str, default='checkpoint/logging', help='path of logging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='optimizer weight decay')
    parser.add_argument('--datapath', type=str, default='data', help='root path of augumented WFLW dataset')
    parser.add_argument('--pretrained', type=str,default='checkpoint/model_weights/weights_epoch_80.pth.tar',help='load checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--savepath', type=str, default='checkpoint/model_weights', help='save checkpoint path')    
    parser.add_argument('--savefreq', type=int, default=5, help="save weights each freq num of epochs")
    parser.add_argument('--logdir', type=str, default='checkpoint/logging', help='logging')    
    parser.add_argument("--lr_patience", default=40, type=int)
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test', help='dataset mode')    
    args = parser.parse_args()
    return args
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()

def main():
    mini_xception = Mini_Xception()
    mini_xception.to(device)

    checkpoint = torch.load(args.pretrained)
    mini_xception.load_state_dict(checkpoint['mini_xception'])
    print(f'\tLoaded checkpoint from {args.pretrained}\n')

    dataset = FER2013(args.datapath, args.mode, transform=transforms.ToTensor())
    print(f'dataset size = {len(dataset)}')
    
    with torch.no_grad():
        for i in range(len(dataset)):

            face, label = dataset[i]
            temp_face = face.squeeze().numpy()
            # print('label',label, 'shape',face.shape)

            face = face.to(device)
            face = torch.unsqueeze(face, 0)
            emotion = mini_xception(face)

            torch.set_printoptions(precision=6)
            softmax = nn.Softmax()
            emotions_soft = softmax(emotion.squeeze()).reshape(-1,1).cpu().detach().numpy()
            emotions_soft = np.round(emotions_soft, 3)
            print(f'softmax {emotions_soft}')

            _, emotion = torch.max(emotion, axis=1)

            temp_face = cv2.resize(temp_face, (200,200))
            cv2.putText(temp_face, get_label_emotion(emotion.squeeze().cpu().item()), (0,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
            cv2.putText(temp_face, get_label_emotion(label.item()), (110,190), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            cv2.imshow('face', temp_face)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break
            print('')


def validate(model, criterion, dataloader, epoch):
    model.eval()
    model.to(device)
    losses = []
    TP = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            mini_batch = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)

            emotions = model(images)
            emotions = torch.squeeze(emotions)
            emotions = emotions.reshape(mini_batch, -1)
            # print(emotions.shape, labels.shape)

            loss = criterion(emotions, labels)            
            print(type(loss))
            losses.append(loss.cpu().item())

            # index of the max value of each sample (shape = (batch,))
            _, indexes = torch.max(emotions, axis=1)
            TruePositive = (indexes == labels).sum().item()
            TP += TruePositive
            # print(indexes.shape, labels.shape)

            print(f'validation loss = {round(loss.item(),3)}')

    val_loss = np.mean(losses).item()
    val_loss = round(val_loss, 3)
    accuracy = round(TP/3500, 3)

    print(f'Accuracy = {accuracy}')
    return val_loss, accuracy

if __name__ == "__main__":
    main()
