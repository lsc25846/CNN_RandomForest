# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:34:03 2022

@author: Bestvision
"""

import numpy as np 
import cv2

from CNN import CNN_layer
import joblib
from SharedImageUseOpenCV import SharedMemory_Image

import argparse

def Test():
    
    SIZE = 64
    
    #--------------------
    im = SharedMemory_Image.Read_Image('test1',10394)
    im = cv2.resize(im, (SIZE, SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = np.expand_dims(im, 0)    
    ###################################################################
    # Normalize pixel values to between 0 and 1
    im = im/255.0

    #############################
    cnn_model,feature_extractor  = CNN_layer.build_CNN()
    cnn_model.load_weights("my_model.h5")

    #RANDOM FOREST
    RF_model = joblib.load("random_forest.joblib")
    #Send test data through same feature extractor process
    #X_test_feature = feature_extractor.predict(x_validation)
    X_test_feature = feature_extractor.predict(im)
    #Now predict using the trained RF model. 
    prediction_RF = RF_model.predict(X_test_feature)
    return prediction_RF    

def MakeArgParser():
    parser = argparse.ArgumentParser()

    #定義每一個參數的名稱、資料型態、預設值、說明等
    
    parser.add_argument('--name', type=str, default='test1', help='The image name in the memory')
    parser.add_argument('--size', type=int, default=10394, help='Image size in the memory')
    #parser.add_argument('--timestamp', type=str, default=None, help='The File Name of Result')

    return parser

if __name__ == '__main__':     
    
    parser = MakeArgParser()
    args = parser.parse_args()
    
    
    cnn_model,feature_extractor  = CNN_layer.build_CNN()
    cnn_model.load_weights("my_model.h5")

    #RANDOM FOREST
    RF_model = joblib.load("random_forest.joblib")
    count = 10
    while(True):
        SIZE = 64
        
        #--------------------
        im = SharedMemory_Image.Read_Image('test1',10394)
        im = cv2.resize(im, (SIZE, SIZE))
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = np.expand_dims(im, 0)    
        ###################################################################
        # Normalize pixel values to between 0 and 1
        im = im/255.0
        
        X_test_feature = feature_extractor.predict(im)
        #Now predict using the trained RF model. 
        prediction_RF = RF_model.predict(X_test_feature)
        print(prediction_RF)
        count= count - 1
        if(count<0):
            break
        
        
    #print(Test())
