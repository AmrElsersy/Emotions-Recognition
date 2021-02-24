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
    mean = np.mean(face)
    std = np.std(face)

    # if black_image
    if int(mean) == 0 and int(std) == 0:
        return face

    face = (face - mean) / std  
    face = face.astype(np.float32)
    # print(f'mean = {mean}, std={std}')
    # normalization will change mean/std but will have overflow in max/min values
    face = np.clip(face, -1, 1)
    # convert from [-1,1] range to [0,1]
    face = (face + 1) / 2
    # face = (face * 255).astype(np.uint8)
    return face.astype(np.float32)

def standerlization(image):
    image = tensor_to_numpy(image)    

    # standerlization .. convert it to 0-1 range
    min_img = np.min(image)
    max_img = np.max(image)
    image = (image - min_img) / (max_img - min_img)
    return image

# https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/

def is_black_image(face):
    # training dataset contains 10 images black & 1 in val dataset
    face = tensor_to_numpy(face)
    mean = np.mean(face)
    std = np.std(face)
    if int(mean) == 0 and int(std) == 0:
        return True
    return False

def normalize_dataset_mode_1(image):
    mean = 0.5077425080522144 
    std = 0.21187228780099732
    image = (image - mean) / std
    return image

def normalize_dataset_mode_255(image):
    mean = 129.47433955331468 
    std = 54.02743338925431
    image = (image - mean) / std
    return image
