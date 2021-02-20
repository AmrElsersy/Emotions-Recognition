import cv2
import time
import dlib
from landmarks_detector import *
import numpy as np


# Abstract class / Interface
class FaceDetectorIface:
    def detect_faces(self, frame):
        raise NotImplementedError

class HaarCascadeDetector(FaceDetectorIface):
    def __init__(self):
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def detect_faces(self, frame):
        faces = self.detector.detectMultiScale(frame)
        return faces

# HOG & SVM
class HogSvmDetector(FaceDetectorIface):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect_faces(self, frame):
        faces = self.detector(frame,1)
        faces = self.convert_faces_format(faces)
        return faces

    def convert_faces_format(self, faces):
        list_faces = []
        for face in faces:
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            list_faces.append((x,y,w,h))
        return list_faces        

class DnnDetector(FaceDetectorIface):
    """
        SSD (Single Shot Detectors) based face detection (ResNet-18 backbone(light feature extractor))
    """
    def __init__(self):
        prototxt = "deploy.prototxt.txt"
        # prototxt = "deploy_lowres.prototxt.txt"
        model_weights = "res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=model_weights)
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

class MtcnnFaceDetector(FaceDetectorIface):
    def __init__(self):
        from mtcnn.mtcnn import MTCNN
        self.detector = MTCNN()
        self.threshold = 0.5
    def detect_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        boxes = []
        for face in faces:
            if face['confidence'] < self.threshold:
                continue

            (x,y,w,h) = face['box']
            boxes.append((x,y,w,h))

        return boxes
# ===================================================
def test(faceDetector, landmarksDetector):
    video = cv2.VideoCapture(0) # 480, 640
    # video = cv2.VideoCapture("4.mp4") # (720, 1280) or (1080, 1920)
    t1 = 0
    t2 = 0
    while video.isOpened():
        _, frame = video.read()

        t2 = time.time()
        fps = round(1/(t2-t1))
        t1 = t2

        # faces
        faces = faceDetector.detect_faces(frame)
        face_frame = np.copy(frame)

        for face in faces:
            # landmarks
            landmarks = landmarksDetector.detect_landmarks(frame, face)
            
            # Draw
            (x,y,w,h) = face

            face_frame[y:y+h, x:x+w] = 0
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)
            for (x,y) in landmarks:
                cv2.circle(frame, (x,y), 1, (0,255,0), -1)        
                cv2.circle(face_frame, (x,y), 1, (0,255,0), -1)        
    

        # cv2.putText(frame, str(fps), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.imshow("Video", frame)   
        cv2.imshow("Landmark", face_frame)     
        if cv2.waitKey(1) & 0xff == 27:
            video.release()
            cv2.destroyAllWindows()
            break


# face_detector = HogSvmDetector()
# face_detector = HaarCascadeDetector()
face_detector = DnnDetector()
landmarksDetector = dlibLandmarks()

test(face_detector, landmarksDetector)

