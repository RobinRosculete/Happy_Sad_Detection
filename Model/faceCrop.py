#Face detection using haarcascade from opencv
from tkinter import X
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import os
import shutil
import pywt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report





face_cascade = cv2.CascadeClassifier('/Users/Robin1/Desktop/Projects/Happy_Sad_detector/Model/opncv/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/Robin1/Desktop/Projects/Happy_Sad_detector/Model/opncv/haarcascade_eye.xml')
        
#function purpose to crop an image that contains both face and two eyes
def get_cropped_image_if_eyes(image_path):
    image = cv2.imread(image_path)
    faces = face_cascade.detectMultiScale(image,1.3, 5)
    for (face_x,face_y,face_w,face_h) in faces:
     roi_color = image[face_y:face_y+face_h, face_x:face_x+face_w]
     eyes = eye_cascade.detectMultiScale(roi_color)
     if len(eyes) > 2:
         return roi_color
     else: return None
     
def wavelet2d(image, mode='haar', level=1):
    imageArray = image
    #converting to gray scale
    imageArray = cv2.cvtColor(imageArray, cv2.COLOR_RGB2GRAY)
    
    #convertion values to float
    imageArray = np.float32(imageArray)
    imageArray /= 255
    
    #calculating coefficents
    coefficients = pywt.wavedec2(imageArray, mode,level=level)
    
    #process coefficients
    coefficients_H=list(coefficients)
    coefficients_H[0] *= 0
    
    #recunstructiing image
    imageArray_H = pywt.waverec2(coefficients_H,mode)
    imageArray_H *= 255
    imageArray_H= np.uint8(imageArray_H)
    return imageArray_H
   

path_to_data = '/Users/Robin1/Desktop/Projects/Happy_Sad_detector/Model/Dataset'
path_to_cropped_data = '/Users/Robin1/Desktop/Projects/Happy_Sad_detector/Model/Dataset/Cropped/'

image_directories =[]
for entry in os.scandir(path_to_data):
    if entry.is_dir():
         image_directories.append(entry.path)
         

if os.path.exists(path_to_cropped_data):
  shutil.rmtree(path_to_cropped_data) #delete path
os.mkdir(path_to_cropped_data)

cropped_image_directories = []
person_file_names_dictionery = {} #dictonery 

#Gettig name of image directory (Happy Person or Sad Person face directory)
for image_directorie in image_directories:
     count = 1
     person_face = image_directorie.split('/')[-1] #geeting diclrectory name
   
     person_file_names_dictionery[person_face] = []
     
     for entry in os.scandir(image_directorie):
       
         roi_color = get_cropped_image_if_eyes(entry.path)
         if roi_color is not None:
              cropped_folder = path_to_cropped_data + person_face
              if not os.path.exists(cropped_folder):
                  os.makedirs(cropped_folder)
                  cropped_image_directories.append(cropped_folder)
                      
              cropped_file_name = person_face  + str(count)+ ".png"
              cropped_file_path = cropped_folder + "/" + cropped_file_name
              cv2.imwrite(cropped_file_path,roi_color)
              person_file_names_dictionery[person_face].append(cropped_file_path)
              count+=1
              
class_dictionery = {}
count = 0 
for person_face in person_file_names_dictionery.keys():
    class_dictionery[person_face] = count
    count +=1 
X= [] #model input
Y = []  #model output
#Staking initial image and wavelet image
for person_face, training_file in person_file_names_dictionery.items():
    for training_image in training_file:
        image = cv2.imread(training_image)
        if image is not None:
         saclled_raw_image = cv2.resize(image,(32,32))
         image_har =wavelet2d(image,'db1',5)
         sacalled_image_har = cv2.resize(image_har,(32,32))
         combined_image = np.vstack((saclled_raw_image.reshape(32*32*3,1),sacalled_image_har.reshape(32*32,1)))
         X.append(combined_image)
         Y.append(class_dictionery[person_face])
  
#reshaping  to (40,4096), and converting to flaot
X = np.array(X).reshape(len(X),4096).astype(float)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf',C =100))])
pipe.fit(X_train, Y_train) 
print (classification_report(Y_test, pipe.predict(X_test)))
