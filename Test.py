# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:54:19 2022

@author: Bestvision
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:19:32 2022

@author: Bestvision
"""

import numpy as np 
import glob
import cv2

import os
import seaborn as sns

from CNN import CNN_layer
#from RandomForest import RF_layer
import joblib
#import time

from SharedImageUseOpenCV import SharedMemory_Image

def Test():
    #start = time.time()
    
    SIZE = 64
    """validation_images = []
    validation_labels = [] 
    for directory_path in glob.glob("Images/Validation/*"):
        validation_label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            validation_images.append(img)
            validation_labels.append(validation_label)
    validation_images = np.array(validation_images)
    validation_labels = np.array(validation_labels)"""
    
    #--------------------
    im = SharedMemory_Image.Read_Image('test1',10394)
    im = cv2.resize(im, (SIZE, SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im = np.expand_dims(im, 0)
    #--------------------
    
    #Encode labels from text to integers.
    """from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(validation_labels)
    validiation_labels_encoded = le.transform(validation_labels)"""

    #Split data into test and train datasets (already split but assigning to meaningful convention)
    #x_validation, y_validation = validation_images, validiation_labels_encoded

    ###################################################################
    # Normalize pixel values to between 0 and 1
    #x_validation = x_validation/255.0
    im = im/255.0

    #############################
    cnn_model,feature_extractor  = CNN_layer.build_CNN()

    #print(cnn_model.summary()) 

    ##########################################
    #Train the CNN model
    cnn_model.load_weights("my_model.h5")

    #RANDOM FOREST
    RF_model = joblib.load("random_forest.joblib")
    #Send test data through same feature extractor process
    #X_test_feature = feature_extractor.predict(x_validation)
    X_test_feature = feature_extractor.predict(im)
    #Now predict using the trained RF model. 
    prediction_RF = RF_model.predict(X_test_feature)
    return prediction_RF
    #Inverse le transform to get original label back. 
    """prediction_RF = le.inverse_transform(prediction_RF)

    #Print overall accuracy
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    print ("Accuracy = ", metrics.accuracy_score(validation_labels, prediction_RF))

    #Confusion Matrix - verify accuracy of each class
    cm = confusion_matrix(validation_labels, prediction_RF)
    #print(cm)
    sns.heatmap(cm, annot=True)
    end = time.time()
    
    print("execution time: " + str(round((end - start),2)) + " second")
    return metrics.accuracy_score(validation_labels, prediction_RF)"""
    
if __name__ == '__main__':    
    #m = SharedMemory_Image.Read_Image('test1',10394)
    print(Test())
