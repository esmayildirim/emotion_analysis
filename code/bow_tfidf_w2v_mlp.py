#new file
import code_pipeline
from code_pipeline import bag_of_words_matrix
from code_pipeline import emotion_matrix
from code_pipeline import count_vectorizer
from code_pipeline import tfidf_vectorization
from code_pipeline import word2vec_vectorization
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedKFold
# from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers import BatchNormalization
from keras.layers import Activation
from mlsmote_lib import MLSMOTE
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
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



'''
class WeightCapture(Callback):
    "Capture the weights of each layer of the model"
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weights = []
        self.epochs = []
 
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch) # remember the epoch axis
        weight = {}
        for layer in self.model.layers:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[0]
            weight[name] = layer.weights[0].numpy()
        self.weights.append(weight)

'''
def model_mlp(X, y):
    model = Sequential()
    model.add(Dense(512, input_dim = X.shape[1], kernel_initializer= 'he_uniform',bias_initializer='he_uniform', activation = 'relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(0.3))
    #model.add(Dense(512, kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform', activation ='relu'))
    #model.add(Dropout(0.3))
    #model.add(Dense(512, kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform', activation ='relu'))
    #model.add(Dropout(0.3))
    #model.add(Dense(512, kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform', activation ='relu'))
    #model.add(Dropout(0.35))
    
    model.add((Dense(y.shape[1], activation='sigmoid')))
    opt = Adam(learning_rate=0.0001)
    model.compile(loss = 'binary_crossentropy', optimizer=opt, metrics = ['accuracy'])
    return model

def confusion_matrix(matrix, yhat, ytest ):
    if yhat == ytest == 1:
        matrix[0]+=1
    elif yhat == ytest == 0:
        matrix[1]+=1
    elif yhat == 1 and ytest == 0:
        matrix[2]+=1
    elif yhat == 0 and ytest == 1:
        matrix[3]+=1
        
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
def plotweight(capture_cb):
    "Plot the weights' mean and s.d. across epochs"
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 10))
    ax[0].set_title("Mean weight")
    for key in capture_cb.weights[0]:
        ax[0].plot(capture_cb.epochs, [w[key].mean() for w in capture_cb.weights], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in capture_cb.weights[0]:
        ax[1].plot(capture_cb.epochs, [w[key].std() for w in capture_cb.weights], label=key)
    ax[1].legend()
    plt.show()
    
def emotion_balance(y):
    emotion_count= [0,0,0,0,0]
    for i in range(y.shape[0]):
        if y[i][0] == 1:
           emotion_count[0] += 1
        if y[i][1] == 1:
           emotion_count[1] += 1
        if y[i][2] == 1:
           emotion_count[2] += 1
        if y[i][3] == 1:
           emotion_count[3] += 1
        if y[i][4] == 1:
           emotion_count[4] += 1
    #print(emotion_count)

if __name__ == '__main__':
    file = "../stemmed_TR.xlsx"
    #BOW vs TFIDF
    #X = count_vectorizer(file, False, True, 2)
    X = tfidf_vectorization(file, max_f = None)
    y = emotion_matrix(file)
    
    #W2V
    '''
    #word2vector with google data set
    modelw2v = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary = True)
    MAX_LENGTH = 250
    #create a MAX_LENGTH by 300 matrix for each text
    dataset = pd.read_excel(file)
    vectorized_headlines = vectorize_data(dataset["Text"], modelw2v, 300, MAX_LENGTH)
    X = np.array(vectorized_headlines)# convert into numpy array
    print(X.shape)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    '''
            
    
    
    #multiple runs lists
    anger_list = [] # each element is a list of acc, prec, recall
    fear_list = []
    sadness_list = []
    disgust_list = []
    joy_list = []
    RANGE = 5
    kf = KFold(n_splits=5, shuffle = False)

    for train_indices, test_indices in kf.split(X):
        #X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]
        emotion_balance(y_train)
        X_train, y_train = MLSMOTE(X_train, y_train, n_neighbors=5, target_ratio=1.0)
        #print("After MLSMOTE")
        emotion_balance(y_train)
        model_train = model_mlp(X,y)
        #capture_cb = WeightCapture(model_train)
        #capture_cb.on_epoch_end(-1)
        
        weights_file = 'tfidf_best_weights.keras'
        checkpoint = ModelCheckpoint(weights_file,
                                     monitor='val_loss',
                                     verbose=0,
                                     mode='min',
                                     save_best_only=True)
        history = model_train.fit(X_train, y_train, epochs=200,callbacks = [checkpoint],  verbose=0, validation_split=0.2)
        #plotweight(capture_cb)
        '''
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper right')
        plt.show()
        '''
        model_train.load_weights(weights_file)
        yhat = model_train.predict(X_test, verbose = 0)
        # round probabilities to class labels
        yhat = yhat.round()
        # calculate accuracy
    
        anger = [0,0,0,0] # TP, TN, FP, FN
        fear = [0,0,0,0]
        disgust = [0,0,0,0]
        joy = [0,0,0,0]
        sadness = [0,0,0,0]
    
        for j in range(len(yhat)):
            confusion_matrix(anger, yhat[j][0], y_test[j][0])
            confusion_matrix(fear, yhat[j][1], y_test[j][1])
            confusion_matrix(sadness, yhat[j][2], y_test[j][2])
            confusion_matrix(disgust, yhat[j][3], y_test[j][3])
            confusion_matrix(joy, yhat[j][4], y_test[j][4])
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
