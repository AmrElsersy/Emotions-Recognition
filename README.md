
# Real Time Emotion Recognition (mini-Xception)

A Pytorch implementation of "Real-time Convolutional Neural Networks for Emotion and Gender Classification" (mini-Xception) [Paper](https://arxiv.org/pdf/1710.07557.pdf)


## Demo

![1](https://user-images.githubusercontent.com/35613645/116496324-162c3c00-a8a5-11eb-9701-414406b745d1.gif)


#### mini-Xception
<img src="https://user-images.githubusercontent.com/35613645/113336812-365cef80-9327-11eb-992a-f88bf18db550.png" width="400" height="400">


#### How to Install
```
 $ pip3 install -r requirements.txt
 ```
 Note that it can be run on lower versions of Pytorch so replace the versions with yours

#### install opencv & dnn from source (optional)
Both opencv dnn & haar cascade are used for face detection, if you want to use haar cascade you can skip this part.

install dependencies 
```
$ sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt-get install libgtk-3-dev
$ sudo apt-get install libatlas-base-dev gfortran
```
Download & install opencv with contrib modules from source
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.2.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.2.0.zip
unzip opencv.zip
unzip opencv_contrib-4.2.0.zip
mkdir -p build && cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.2.0/modules ../opencv
cmake --build .
```
if you have any problems, refere to [Install opencv with dnn from source](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

if you **don't** want to use **dnn** modules just setup opencv with regular way
```
sudo apt-get install python3-opencv
```

#### Run Camera Demo
##### Live camera demo 
```python
$ python3 camera_demo.py

# add '--haar' option if you want to use Haar cascade detector instead of dnn opencv face detector
$ python3 camera_demo.py --haar
```

### Test 
##### image test
```
# replace $PATH_TO_IMAGE with your relative(or global) path to the image 
$ python3 camera_demo.py --image --path PATH_TO_IMAGE
```
##### video test
```
$ python3 camera_demo.py --path PATH_TO_VIDEO
```


#### Face Preprocessing
- Histogram Equalization for iliumination normalization 
- Face Alignment using dlib landmarks
##### Demo

![2](https://user-images.githubusercontent.com/35613645/116496346-22b09480-a8a5-11eb-9715-cefb41d221cc.gif)


### FER2013 Dataset
The data consists of **48x48 pixel grayscale** images of faces. and their emotion shown in the facial expression in to one of seven categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral), The training set consists of **28,709 examples**. The public test set consists of **3,589 examples**.

[Download FER2013](https://www.kaggle.com/deadskull7/fer2013)
- create a folder called "data/" in project root
- put the "fer2013.csv" in it


#### Visualize dataset
Visualize dataset examples with annotated landmarks & head pose 
```cmd
# add '--mode' option to determine the dataset to visualize
$ python3 visualization.py
```
#### Tensorboard 
Take a wide look on dataset examples using tensorboard
```
$ python3 visualization.py --tensorboard
$ tensorboard --logdir checkpoint/tensorboard
```
![Screenshot 2021-04-01 20:05:42](https://user-images.githubusercontent.com/35613645/113335766-aff3de00-9325-11eb-8c07-66379e53a65d.png)



#### Testing
```
$ python3 test.py
```

#### Training 
```
$ python3 train.py
```
#### Evaluation
```
$ python3 train.py --evaluate
```
will show the confision matrix

![Screenshot 2021-04-01 20:13:14](https://user-images.githubusercontent.com/35613645/113336651-04e42400-9327-11eb-8aa1-d52d78eb0ad5.png)

#### Folder structure    
    ├── model					# model's implementation
    ├── data					# data folder contains FER2013 dataset
    ├── train					# train on FER2013 dataset 
    ├── test					# test on 1 example
    ├── face_detector			# contain the code of face detection (dnn & haar-cascade)
    ├── face_alignment			# contain the code of face alignment using dlib landmarks


#### Refrences
Deep Learning on Facial Expressions Survey
- https://arxiv.org/pdf/1804.08348.pdf

ilimunation normalization (histogram / GCN / Local Norm)
- https://www.sciencedirect.com/science/article/pii/S1877050917320860

Tensorflow Implementation
- https://github.com/oarriaga/face_classification/tree/master

Inception (has some used blocks)
- https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202

Xception
- https://towardsdatascience.com/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568

Pytorch GlobalAvgPooling
- https://paperswithcode.com/method/global-average-pooling


