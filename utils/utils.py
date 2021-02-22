import numpy as np

def get_label_emotion(label : int) -> str:
    label_emotion_map = { 
        0: 'Angry',
        1: 'Disgust', 
        2: 'Fear', 
        3: 'Happy', 
        4: 'Sad', 
        5: 'Surprise', 
        6: 'Neutral'        
    }
    return label_emotion_map[label]

def standerlization(face):
    face = (face - np.mean(face))/ np.std(face)
    face = (face * 255).astype(np.uint8)
    return face