"""
Author: Amr Elsersy
email: amrelsersay@gmail.com
-----------------------------------------------------------------------------------
Description: utils functions
"""
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

def tensor_to_numpy(image):
    if type(image) != np.ndarray:
        return image.cpu().squeeze().numpy()
    return image

def normalization(face):
    face = tensor_to_numpy(face)
    # [-1,1] range
    face = (face - np.mean(face)) / np.std(face)
    # normalization will change mean/std but will have overflow in max/min values
    face = np.clip(face, -1, 1)
    # convert from [-1,1] range to [0,1]
    face = (face + 1.0) / 2.0
    # face = (face * 255).astype(np.uint8)
    return face

def standerlization(image):
    # standerlization .. convert it to 0-1 range
    image = tensor_to_numpy(image)    
    min_img = np.min(image)
    max_img = np.max(image)
    image = (image - min_img) / (max_img - min_img)
    return image

# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/