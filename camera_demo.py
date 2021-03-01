"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: Live Camera Demo using opencv dnn face detection & Emotion Recognition
"""
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from numpy.lib.type_check import imag
import torch
from torch.functional import norm
import torchvision.transforms.transforms as transforms
from face_detector.face_detector import DnnDetector, HaarCascadeDetector

from model.model import Mini_Xception
from utils import get_label_emotion, normalization, histogram_equalization, standerlization


sys.path.insert(1, 'face_detector')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_preprocess(input_face):
    histogram = histogram_equalization(input_face)
    normalized = normalization(input_face)
    normalized_histogram = normalization(histogram)

    shape = (200,200)
    histogram = cv2.resize(histogram, shape)
    normalized = cv2.resize(normalized, shape)
    normalized_histogram = cv2.resize(normalized_histogram, shape)

    cv2.imshow('normal', normalized)
    cv2.imshow('normal_histogram', normalized_histogram)
    cv2.imshow('histogram', histogram)


def main(args):
    # Model
    mini_xception = Mini_Xception().to(device)
    mini_xception.eval()

    # Load model
    checkpoint = torch.load(args.pretrained, map_location=device)
    mini_xception.load_state_dict(checkpoint['mini_xception'])

    # Face detection
    root = 'face_detector'
    face_detector = None
    if args.haar:
        face_detector = HaarCascadeDetector(root)
    else:
        face_detector = DnnDetector(root)

    video = cv2.VideoCapture(0) # 480, 640
    # video = cv2.VideoCapture("face_detector/1.mp4") # (720, 1280) or (1080, 1920)
    t1 = 0
    t2 = 0
    print('video.isOpened:', video.isOpened())
    while video.isOpened():
        _, frame = video.read()

        # time
        t2 = time.time()
        fps = round(1/(t2-t1))
        t1 = t2

        # faces
        faces = face_detector.detect_faces(frame)

        for face in faces:
            (x,y,w,h) = face

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

            # preprocessing
            input_face = frame[y:y+h, x:x+w]
            input_face = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
            input_face = cv2.resize(input_face, (48,48))
        
            # test_preprocess(input_face)
            input_face = histogram_equalization(input_face)
            # cv2.imshow('face', input_face)

            input_face = transforms.ToTensor()(input_face).to(device)
            input_face = torch.unsqueeze(input_face, 0)

            with torch.no_grad():
                input_face = input_face.to(device)
                t = time.time()
                emotion = mini_xception(input_face)
                print(f'\ntime={(time.time()-t) * 1000 } ms')

                torch.set_printoptions(precision=6)
                softmax = torch.nn.Softmax()
                emotions_soft = softmax(emotion.squeeze()).reshape(-1,1).cpu().detach().numpy()
                emotions_soft = np.round(emotions_soft, 3)
                for i, em in enumerate(emotions_soft):
                    em = round(em.item(),3)
                    print(f'{get_label_emotion(i)} : {em}')

                emotion = torch.argmax(emotion)                
                percentage = round(emotions_soft[emotion].item(), 2)
                emotion = emotion.squeeze().cpu().detach().item()
                emotion = get_label_emotion(emotion)

                cv2.putText(frame, emotion, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
                cv2.putText(frame, str(percentage), (x + w - 40,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

        cv2.putText(frame, str(fps), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.imshow("Video", frame)   
        if cv2.waitKey(1) & 0xff == 27:
            video.release()
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--haar', action='store_true', help='run the haar cascade face detector')
    parser.add_argument('--pretrained',type=str,default='checkpoint/model_weights/weights_epoch_75.pth.tar' 
                        ,help='load weights')
    parser.add_argument('--head_pose', action='store_true', help='visualization of head pose euler angles')
    args = parser.parse_args()

    main(args)

