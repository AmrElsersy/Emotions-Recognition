from sys import flags
import cv2
import numpy as np
import math
from .dlib_landmarks.landmarks_detector import dlibLandmarks

class FaceAlignment:   
    def __init__(self):
        self.dlib_landmarks = dlibLandmarks()     
    
    def get_face_rotation_angle(self, landmarks):
        right_eye = landmarks[0]
        left_eye = landmarks[1]

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def get_rotation_center(self, landmarks, face_rect):
        (x,y,w,h) = face_rect
        center = (int(x+w/2), int(y+h/2))
        # left_eye = landmarks[1]
        # center = ( ( (left_eye + right_eye) / 2) * (w,h) / 48).astype(np.float32)
        # center = (center[0] + x, center[1] + y)
        return center

    def get_eyes_landmarks(self, _landmarks, face_rect):
        landmarks = np.copy(_landmarks)
        (x,y,w,h) = face_rect
        landmarks -= (x,y)

        # avarage of 2 points eye
        right_eye = ((landmarks[0] + landmarks[1])//2).astype(np.uint8)
        left_eye  = ((landmarks[2] + landmarks[3])//2).astype(np.uint8)

        landmarks = np.array([right_eye, left_eye]).astype(np.uint8)
        # scale 
        landmarks = (landmarks / (w,h) * 48).astype(np.float32)
        return landmarks

    def get_new_rect(self, face_rect, center, angle, max_shape):
        (x,y,w,h) = face_rect
        max_h, max_w = max_shape
        (xc,yc) = center

        angle_percentage = abs(angle) / 90
        # assume that percentage of w/h is usually around 7/10
        # calculate new h & w
        new_h = h * (1 + 0.25 * angle_percentage)
        new_w = 0.75 * new_h

        x1 = int(xc - new_w/2)
        y1 = int(yc - new_h/2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        
        x2 = int(x1 + new_w)
        y2 = int(y1 + new_h)
        x2 = min(x2, max_w)
        y2 = min(y2, max_h)

        return ((x1,y1), (x2,y2))

    def frontalize_face(self, face_rect, _frame):
        # get the landmarks
        landmarks = self.dlib_landmarks.detect_landmarks(_frame, face_rect)
        # avarage the eye's points to get 1 point for each eye
        filtered_landmarks = self.get_eyes_landmarks(landmarks, face_rect)
        # get the angle of the line between the 2 eyes
        angle = self.get_face_rotation_angle(filtered_landmarks)
        # rotation face center
        center = self.get_rotation_center(filtered_landmarks, face_rect)
        # rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # rotate the whole frame
        frame = np.copy(_frame)
        shape = (frame.shape[1], frame.shape[0])
        frame = cv2.warpAffine(frame, M, shape, flags=cv2.INTER_LINEAR)

        # calculate the new h,w to wrap the whole face
        rect = self.get_new_rect(face_rect, center, angle, frame.shape[0:2])
        ((x1,y1), (x2,y2)) = rect

        # crop the face & convert to gray
        face = frame[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return face


