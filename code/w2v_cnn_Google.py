#new file
import code_pipeline
from code_pipeline import bag_of_words_matrix
from code_pipeline import emotion_matrix
from code_pipeline import normalization
from code_pipeline import translate
from code_pipeline import english_stopword_removal_lemmatization_stemming
from code_pipeline import turkish_stopword_removal_lemmatization_stemming
import numpy as np
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedKFold
# from tensorflow import keras
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


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

def cnn_model_arch(X_train, y_train,  VECTOR_SIZE = 300, MAX_LENGTH = 200):
    FILTERS = 16 #feature maps
    KERNEL_SIZE = 7
    HIDDEN_LAYER_NODES = 128
    DROPOUT_PROB = 0.35
    
    
    cnn_model = Sequential()
    cnn_model.add(Conv1D(FILTERS, KERNEL_SIZE, padding = 'same', strides = 1, activation = 'relu', input_shape=(MAX_LENGTH, VECTOR_SIZE)))
    #cnn_model.add(Conv1D(FILTERS, KERNEL_SIZE, padding = 'same', strides = 1, activation = 'relu'))
    #add more conv layers
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(HIDDEN_LAYER_NODES,activation = 'relu'))
    cnn_model.add(Dropout(DROPOUT_PROB))
    #add more dense and dropout layer
    cnn_model.add(Dense(HIDDEN_LAYER_NODES,activation = 'relu'))
    cnn_model.add(Dropout(DROPOUT_PROB))
    #output layer remains the same always
    cnn_model.add(Dense(5, activation = 'sigmoid'))
    print(cnn_model.summary())
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
if __name__ == '__main__':
    #normalization("./dataset_labeled_OR.xlsx", "./normalized_TR.xlsx")
    #translate("./normalized_TR.xlsx", "./normalized_ENG.xlsx")
    #english_stopword_removal_lemmatization_stemming("./normalized_ENG.xlsx", "./stemmed_ENG.xlsx", "./lemmatized_ENG.xlsx")
    #turkish_stopword_removal_lemmatization_stemming("./normalized_TR.xlsx", "./stemmed_TR.xlsx", "./lemmatized_TR.xlsx")
    
    file = "lemmatized_ENG.xlsx"
    dataset = pd.read_excel(file)
    text = dataset["Text"] # Texts that don't have periods/punctuation so they're just sentences
    print("Length of Texts:", len(text))
    
    #find average max min number of words in texts so that we can cut each text to be of the same length

    Length = []
    for paragraph in text:
        Length.append(len(paragraph.split()))
    print(max(Length),min(Length),mean(Length))
    print(Length)
    #load the google news word to vector model
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
    MAX_LENGTH = 500
    #create a MAX_LENGTH by 300 matrix for each text
    vectorized_headlines = vectorize_data(text, model, 300, MAX_LENGTH)
    arr = np.array(vectorized_headlines)# convert into numpy array
    print(arr.shape)
    #load the output matrix
    y = emotion_matrix(file)
    #train the cnn model
    RANGE = 10
    for i in range(RANGE):
    
        X_train, X_test, y_train, y_test = train_test_split(arr,y)
        cnnm = cnn_model_arch(X_train, y_train, 300, MAX_LENGTH)
        opt = Adam(learning_rate = 0.0001)
        cnnm.compile(loss='binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
        history = cnnm.fit(X_train, y_train, epochs=150, batch_size = 50, validation_split = 0.2)
    
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper right')
        plt.show()
    # test the model
        yhat = cnnm.predict(X_test)
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

        print ("Accuracy:\t%.2f %.2f %.2f %.2f %.2f" % (accuracy(anger), accuracy(fear), accuracy(disgust),accuracy(sadness), accuracy(joy)))
        print("Precision:\t%.2f %.2f %.2f %.2f %.2f" % (precision(anger), precision(fear), precision(disgust), precision(sadness), precision(joy)))
        print("Recall:\t\t%.2f %.2f %.2f %.2f %.2f" %  (recall(anger), recall(fear), recall(disgust), recall(sadness), recall(joy)))
        print("F1-score:\t%.2f %.2f %.2f %.2f %.2f" %  (f1_score(anger), f1_score(fear), f1_score(disgust), f1_score(sadness), f1_score(joy)))
   
