#new file
import code_pipeline
from code_pipeline import bag_of_words_matrix
from code_pipeline import emotion_matrix
from code_pipeline import normalization
from code_pipeline import translate
from code_pipeline import english_stopword_removal_lemmatization_stemming
from code_pipeline import turkish_stopword_removal_lemmatization_stemming
from code_pipeline import tfidf_vectorization

import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedKFold
# from tensorflow import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D
from code_pipeline import count_vectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from mlsmote_lib import MLSMOTE
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

def vectorize_data(data, model, VECTOR_SIZE = 300, MAX_LENGTH = 200):
    #print(model.key_to_index.keys())
    vectors = []
    padding_vector = [0.0] * VECTOR_SIZE
    vocab = list(model.key_to_index.keys())
    #print(vocab)
    for i, data_point in enumerate(data): #18000 text
        data_point_vectors = []
        count = 0
        tokens = data_point.split() # each texts tokens
        for token in tokens:
            
            if count >= MAX_LENGTH:
                break
            #print(token)
            if token in vocab:
                #print("I am in", token)
                data_point_vectors.append(model[token])
            count = count+1
        if len(data_point_vectors) < MAX_LENGTH:
            to_fill = MAX_LENGTH - len(data_point_vectors)
            for _ in range (to_fill):
                data_point_vectors.append(padding_vector)
        vectors.append(data_point_vectors)
    return vectors

def cnn_model_arch(X_train, y_train):
    FILTERS = 8 #feature maps
    KERNEL_SIZE = 3
    HIDDEN_LAYER_NODES = 128
    DROPOUT_PROB = 0.35
    cnn_model = Sequential()
    cnn_model.add(Conv1D(FILTERS, KERNEL_SIZE, padding = 'same', strides = 1, activation = 'relu', input_shape=(X_train.shape[1],1)))
    #cnn_model.add(Conv1D(FILTERS, KERNEL_SIZE, padding = 'same', strides = 1, activation = 'relu'))
    #add more conv layers
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(HIDDEN_LAYER_NODES,activation = 'relu'))
    cnn_model.add(Dropout(DROPOUT_PROB))
    #add more dense and dropout layer
    #cnn_model.add(Dense(HIDDEN_LAYER_NODES,activation = 'relu'))
    #cnn_model.add(Dropout(DROPOUT_PROB))
    #output layer remains the same always
    cnn_model.add(Dense(5, activation = 'sigmoid'))
    #print(cnn_model.summary())
    return cnn_model


def confusion_matrix(matrix, yhat, ytest ):
    if yhat == ytest == 1:
        matrix[0]+=1
    elif yhat == ytest == 0:
        matrix[1]+=1
    elif yhat == 1 and ytest == 0:
        matrix[2]+=1
    elif yhat == 0 and ytest == 1:
        matrix[3]+=1
        
def f1_score(list):
    return 2 * precision(list) * recall(list) / (precision(list)+recall(list))
def accuracy(list):
    return (list[0]+list[1])/(list[0]+ list[1]+ list[2]+list[3])
def precision(list):
    if (list[0]+list[2]) != 0:
        return list[0]/(list[0]+list[2])
    else: return 0
def recall(list):
    if (list[0]+list[3]) != 0:
        return list[0]/(list[0]+list[3])
    else: return 0
def f1(list):
    return precision(list) * recall(list) * 2 / (precision(list)+recall(list))
if __name__ == '__main__':
    #normalization("./dataset_labeled_OR.xlsx", "./normalized_TR.xlsx")
    #translate("./normalized_TR.xlsx", "./normalized_ENG.xlsx")
    #english_stopword_removal_lemmatization_stemming("./normalized_ENG.xlsx", "./stemmed_ENG.xlsx", "./lemmatized_ENG.xlsx")
    #turkish_stopword_removal_lemmatization_stemming("./normalized_TR.xlsx", "./stemmed_TR.xlsx", "./lemmatized_TR.xlsx")
    
    #COUNT VECTORIZER
    
    
    
    file = "../stemmed_TR.xlsx"
    #BOW vs TFIDF
    #X = count_vectorizer(file, False, True, 2)
    X = tfidf_vectorization(file, max_f = None)

    #X.reshape(X.shape[0], X.shape[1], 1)
    print(X.shape)

    y = emotion_matrix(file)
    #train the cnn model
    kf = KFold(n_splits=5, shuffle = False)
    anger_list = [] # each element is a list of acc, prec, recall
    fear_list = []
    sadness_list = []
    disgust_list = []
    joy_list = []
    RANGE = 5
    for train_indices, test_indices in kf.split(X):
    
        X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        X_train, y_train = MLSMOTE(X_train, y_train, n_neighbors=5, target_ratio=1.0)
        cnnm = cnn_model_arch(X_train, y_train)
        opt = Adam(learning_rate = 0.0001)
        cnnm.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
        weights_file = 'bowcnn_best_weights.keras'
        checkpoint = ModelCheckpoint(weights_file,
                                     monitor='val_loss',
                                     verbose=0,
                                     mode='min',
                                     save_best_only=True)
        history = cnnm.fit(X_train,
                            y_train,
                            epochs=1000,
                            callbacks = [checkpoint],
                            batch_size = 50,
                            verbose = 0,
                            validation_split = 0.2)
    
        #plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        #plt.title('accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train','val'], loc='upper right')
        #plt.show()
    # test the model
        cnnm.load_weights(weights_file)
        yhat = cnnm.predict(X_test, verbose = 0)
        # round probabilities to class labels
        yhat = yhat.round()
    #calculate performance metrics
        anger = [0,0,0,0] # TP, TN, FP, FN
        fear = [0,0,0,0]
        disgust = [0,0,0,0]
        joy = [0,0,0,0]
        sadness = [0,0,0,0]

        for i in range(len(yhat)):
            confusion_matrix(anger, yhat[i][0], y_test[i][0])
            confusion_matrix(fear, yhat[i][1], y_test[i][1])
            confusion_matrix(sadness, yhat[i][2], y_test[i][2])
            confusion_matrix(disgust, yhat[i][3], y_test[i][3])
            confusion_matrix(joy, yhat[i][4], y_test[i][4])

        print("RUN ")
        print ("Accuracy:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy(anger), accuracy(fear), accuracy(disgust),accuracy(sadness), accuracy(joy)))
        print("Precision:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (precision(anger), precision(fear), precision(disgust), precision(sadness), precision(joy)))
        print("Recall:\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %  (recall(anger), recall(fear), recall(disgust), recall(sadness), recall(joy)))
        print("F1-score:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %  (f1(anger), f1(fear), f1(disgust), f1(sadness), f1(joy)))
        print("F1-average:\t%.3f" % ((f1(anger)+ f1(fear)+ f1(disgust)+ f1(sadness)+ f1(joy))/5))
        anger_list.append([accuracy(anger), precision(anger), recall(anger), f1(anger)])
        fear_list.append([accuracy(fear), precision(fear), recall(fear), f1(fear)])
        sadness_list.append([accuracy(sadness), precision(sadness), recall(sadness), f1(sadness)])
        disgust_list.append([accuracy(disgust), precision(disgust), recall(disgust), f1(disgust)])
        joy_list.append([accuracy(joy), precision(joy), recall(joy), f1(joy)])
        
    #calculate means
    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    f1_mean = 0
    f1_total_mean = 0
    print("\tAccuracy\tPrecision\tRecall\tF1_score")
    for k in range(RANGE):
        accuracy_mean += anger_list[k][0]
        precision_mean += anger_list[k][1]
        recall_mean += anger_list[k][2]
        f1_mean += anger_list[k][3]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    f1_mean /= RANGE
    print("Anger\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy_mean, precision_mean, recall_mean, f1_mean))
    f1_total_mean += f1_mean
    #complete for the other emotions
    
    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    f1_mean = 0
    for k in range(RANGE):
        accuracy_mean += fear_list[k][0]
        precision_mean += fear_list[k][1]
        recall_mean += fear_list[k][2]
        f1_mean += fear_list[k][3]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    f1_mean /= RANGE
    print("Fear\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy_mean, precision_mean, recall_mean, f1_mean))
    f1_total_mean += f1_mean

    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    f1_mean = 0
    for k in range(RANGE):
        accuracy_mean += sadness_list[k][0]
        precision_mean += sadness_list[k][1]
        recall_mean += sadness_list[k][2]
        f1_mean += sadness_list[k][3]

    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    f1_mean /= RANGE
    print("Sadness\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy_mean, precision_mean, recall_mean, f1_mean))
    f1_total_mean += f1_mean


    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    f1_mean = 0
    for k in range(RANGE):
        accuracy_mean += disgust_list[k][0]
        precision_mean += disgust_list[k][1]
        recall_mean += disgust_list[k][2]
        f1_mean += disgust_list[k][3]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    f1_mean /=RANGE
    print("Disgust\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy_mean, precision_mean, recall_mean, f1_mean))
    f1_total_mean += f1_mean

    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    f1_mean = 0
    for k in range(RANGE):
        accuracy_mean += joy_list[k][0]
        precision_mean += joy_list[k][1]
        recall_mean += joy_list[k][2]
        f1_mean += joy_list[k][3]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    f1_mean /= RANGE
    f1_total_mean += f1_mean

    print("Joy\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy_mean, precision_mean, recall_mean, f1_mean))
    print("F1_mean\t%.3f" % (f1_total_mean/5))

