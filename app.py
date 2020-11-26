# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:12:26 2020

@author: User
"""

from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__, template_folder='template')

@app.route("/")
def index():
    return render_template('index.html');

@app.route('/background_process_test')
def background_process_test():
    print ("Hello World")
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
            model = load_model('facefeatures_new_model.h5')
            result = model.predict(test_image)

            if result[0][0] == 1:
                prediction = 'dog'
                #return [{ "image" : prediction}]
                print('1 found')
                return jsonify({'index': 0})
            else:
                prediction = 'cat'
                #return [{ "image" : prediction}]
                print('2 found')
                return jsonify({'index': 0})
            
            notFounding = False
    camera.release()

    
    
    

if __name__ == '__main__': 
    app.run() 