"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Testing
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
from dataset import FER2013
from utils import get_label_emotion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
    parser.add_argument('--datapath', type=str, default='data', help='root path of augumented WFLW dataset')
    parser.add_argument('--resume', action='store_true', help='resume from pretrained path specified in prev arg')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'val'], default='test', help='dataset mode')    
    parser.add_argument('--pretrained', type=str,default='checkpoint/model_weights/weights_epoch_75.pth.tar')
    args = parser.parse_args()
    return args
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()

def main():
    mini_xception = Mini_Xception()
    mini_xception.to(device)
    mini_xception.eval()

    checkpoint = torch.load(args.pretrained, map_location=device)
    mini_xception.load_state_dict(checkpoint['mini_xception'], strict=False)
    print(f'\tLoaded checkpoint from {args.pretrained}\n')

    dataset = FER2013(args.datapath, args.mode, transform=transforms.ToTensor())
    print(f'dataset size = {len(dataset)}')
    
    with torch.no_grad():
        for i in range(len(dataset)):
            face, label = dataset[i]
            temp_face = face.squeeze().numpy()

            face = face.to(device)
            face = torch.unsqueeze(face, 0)
            emotion = mini_xception(face)

            # torch.set_printoptions(precision=6)
            # softmax = nn.Softmax()
            # emotions_soft = softmax(emotion.squeeze()).reshape(-1,1).cpu().detach().numpy()
            # emotions_soft = np.round(emotions_soft, 3)
            # for i, em in enumerate(emotions_soft):
            #     em = round(em.item(),3)
            #     print(f'{get_label_emotion(i)} : {em}')
            # # print(f'softmax {emotions_soft}')

            _, emotion = torch.max(emotion, axis=1)

            temp_face = cv2.resize(temp_face, (200,200))
            cv2.putText(temp_face, get_label_emotion(emotion.squeeze().cpu().item()), (0,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
            cv2.putText(temp_face, get_label_emotion(label.item()), (110,190), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0))
            cv2.imshow('face', temp_face)

            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main()
