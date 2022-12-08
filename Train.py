# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:45:47 2022

@author: Bestvision
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2

import os
import seaborn as sns

from CNN import CNN_layer
from RandomForest import RF_layer
import joblib

def train():
    SIZE = 64

    train_images = []
    train_labels = [] 
    for directory_path in glob.glob("Images/Train/*"):
        label = directory_path.split("\\")[-1]
        print(label)
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            train_images.append(img)
            train_labels.append(label)
            
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)


    # test
    test_images = []
    test_labels = [] 
    for directory_path in glob.glob("Images/Test/*"):
        fruit_label = directory_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            test_images.append(img)
            test_labels.append(fruit_label)
            
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    validation_images = []
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
    validation_labels = np.array(validation_labels)

    #Encode labels from text to integers.
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    le.fit(train_labels)
    train_labels_encoded = le.transform(train_labels)
    le.fit(validation_labels)
    validiation_labels_encoded = le.transform(validation_labels)

    #Split data into test and train datasets (already split but assigning to meaningful convention)
    x_train, y_train, x_test, y_test, x_validation, y_validation = train_images, train_labels_encoded, test_images, test_labels_encoded, validation_images, validiation_labels_encoded

    ###################################################################
    # Normalize pixel values to between 0 and 1
    x_train, x_test, x_validation = x_train / 255.0, x_test / 255.0, x_validation/255.0

    #One hot encode y values for neural network. 
    from tensorflow.keras.utils import to_categorical
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    #############################
    cnn_model,feature_extractor  = CNN_layer.build_CNN()

    print(cnn_model.summary())
    
    #Start trainning 開始訓練CNN
    
    history = cnn_model.fit(x_train, y_train_one_hot, epochs=250, validation_data = (x_test, y_test_one_hot))
    
    #儲存權重參數
    cnn_model.save_weights("my_model.h5")
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    prediction_NN = cnn_model.predict(x_test)
    prediction_NN = np.argmax(prediction_NN, axis=-1)
    prediction_NN = le.inverse_transform(prediction_NN)


    #Confusion Matrix - verify accuracy of each class
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, prediction_NN)
    print(cm)
    sns.heatmap(cm, annot=True)
    
    #Random Forest 訓練
    #Now, let us use features from convolutional network for RF
    X_for_RF = feature_extractor.predict(x_train) #This is out X input to RF
    RF_model = RF_layer.build_RF(89,42)
    # Train the model on training data
    RF_model.fit(X_for_RF, y_train)
    
    #
    joblib.dump(RF_model, "random_forest.joblib")
    
    loaded_rf = joblib.load("random_forest.joblib")
    
    X_test_feature = feature_extractor.predict(x_validation)
    
    prediction_RF = loaded_rf.predict(X_test_feature)
    #Inverse le transform to get original label back. 
    prediction_RF = le.inverse_transform(prediction_RF)

    #Print overall accuracy
    from sklearn import metrics
    print ("Accuracy = ", metrics.accuracy_score(validation_labels, prediction_RF))

    #Confusion Matrix - verify accuracy of each class
    cm = confusion_matrix(validation_labels, prediction_RF)
    #print(cm)
    sns.heatmap(cm, annot=True)
    
    
if __name__ == '__main__':    
    train()
   