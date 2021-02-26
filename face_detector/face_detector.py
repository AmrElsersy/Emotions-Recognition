import cv2
import numpy as np
import os

# Abstract class / Interface
class FaceDetectorIface:
    def detect_faces(self, frame):
        raise NotImplementedError

class HaarCascadeDetector(FaceDetectorIface):
    def __init__(self, root=None):
        self.path = "haarcascade_frontalface_default.xml"
        if root:
            self.path = os.path.join(root, self.path)

        self.detector = cv2.CascadeClassifier(self.path)

    def detect_faces(self, frame):
        faces = self.detector.detectMultiScale(frame)
        return faces

class DnnDetector(FaceDetectorIface):
    """
        SSD (Single Shot Detectors) based face detection (ResNet-18 backbone(light feature extractor))
    """
    def __init__(self, root=None):
        self.prototxt = "deploy.prototxt.txt"
        self.model_weights = "res10_300x300_ssd_iter_140000.caffemodel"

        if root:
            self.prototxt = os.path.join(root, self.prototxt)
            self.model_weights = os.path.join(root, self.model_weights)

        self.detector = cv2.dnn.readNetFromCaffe(prototxt=self.prototxt, caffeModel=self.model_weights)
        self.threshold = 0.5 # to remove weak detections

    def detect_faces(self,frame):
        h = frame.shape[0]
        w = frame.shape[1]

        # required preprocessing(mean & variance(scale) & size) to use the dnn model
        """
            Problem of not detecting small faces if the image is big (720p or 1080p)
            because we resize to 300,300 ... but if we use the original size it will detect right but so slow
        """
        resized_frame = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, resized_frame.shape[0:2], (104.0, 177.0, 123.0))
        # detect
        self.detector.setInput(blob)
        detections = self.detector.forward()
        faces = []
        # shape 2 is num of detections
        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2] 
            if confidence < self.threshold:
                continue

            # model output is percentage of bbox dims
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")
            (x1,y1, x2,y2) = box

            # x,y,w,h
            faces.append((x1,y1,x2-x1,y2-y1))
            # print(confidence)
        return faces
