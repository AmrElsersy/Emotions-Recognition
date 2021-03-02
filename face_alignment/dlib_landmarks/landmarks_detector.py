import dlib
import numpy as np
from enum import Enum
import os

class LandmarksDetectorIface:
    def detect_landmarks(self, frame, rect):
        raise NotImplementedError    
    def convert_to_numpy(self,landmarks):
        raise NotImplementedError
    
class dlibLandmarks(LandmarksDetectorIface):

    def __init__(self, root='face_alignment/dlib_landmarks'):
        self.path = "shape_predictor_5_face_landmarks.dat"
        self.path = os.path.join(root, self.path)
        self.detector = dlib.shape_predictor(self.path)

    def convert_to_numpy(self, landmarks):
        num_landmarks = 5
        coords = np.zeros((num_landmarks, 2), dtype=np.int)
        for i in range(num_landmarks):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        return coords

    def detect_landmarks(self, frame, rect):
        # landmarks detection accept only dlib rectangles to operate on
        if type(rect) != dlib.rectangle:
            (x,y,w,h) = rect
            rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

        # convert from dlib style to numpy style
        landmarks = self.detector(frame, rect)
        return self.convert_to_numpy(landmarks)
