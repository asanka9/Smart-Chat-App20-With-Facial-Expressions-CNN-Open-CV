# -*- coding: utf-8 -*-

"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np


model= load_model('facefeatures_new_model.h5')
print('Hiiii')
imagename = 'happy.jpg'
test_image = image.load_img(imagename, target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
     prediction = 'dog'
     #return [{ "image" : prediction}]
     print('Dog')
else:
    prediction = 'cat'
    #return [{ "image" : prediction}]
    print('Cat')
"""
    
    

import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np


model= load_model('facefeatures_new_model.h5')
camera = cv2.VideoCapture(0)
notFounding = True
while notFounding:
    return_value, image_ = camera.read()
    cv2.imwrite('cropped.png', image_)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('cropped.png')
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        print('88888888888')
        img = img[y:y+h, x:x+w]
        cv2.imwrite('abc.png', img)
        imagename = 'abc.png'
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'dog'
            #return [{ "image" : prediction}]
            print('Angry')
        else:
            prediction = 'cat'
            #return [{ "image" : prediction}]
            print('Happy')
            
        notFounding = False
    
camera.release()


"""
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('openc3232v.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if faces is ():
    print('999999999999999')

for (x,y,w,h) in faces:
    print('88888888888')
    img = img[y:y+h, x:x+w]
    cv2.imwrite('cropped.png', img)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
