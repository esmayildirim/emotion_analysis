import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import code_pipeline
from code_pipeline import bag_of_words_matrix
from code_pipeline import emotion_matrix
from code_pipeline import count_vectorizer
from code_pipeline import tfidf_vectorization
from code_pipeline import word2vec_vectorization
from mlsmote_lib import MLSMOTE
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
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

def emotion_matrix(file): #416 by 5 matrix
    dataset = pd.read_excel(file)
    Text = dataset["Text"]
    anger = dataset["Anger"]
    fear = dataset["Fear"]
    sadness = dataset["Sadness"]
    disgust = dataset["Disgust"]
    joy = dataset["Joy"]
    outerList = []
    #Try to make a list [anger1, fear1, sadness1, disgust1, joy1], [anger2, fear2, sadness2, disgust2, joy2], ... [anger416, fear416, sadness416, disgust416, joy416]
    for i in range(len(Text)):
        innerList = []
        innerList.append(anger[i])
        innerList.append(fear[i])
        innerList.append(sadness[i])
        innerList.append(disgust[i])
        innerList.append(joy[i])
        outerList.append(innerList)

    Emotions_array = np.array(outerList)
    print("Shape of Matrix: ", np.shape(Emotions_array))
    print(Emotions_array) #416 rows by 5 columns
    return(Emotions_array)

file_in = "../lemmatized_TR.xlsx"
labels = emotion_matrix(file_in)

#BAG OF WORDS
#MIN_DF = 3
#features = count_vectorizer(file_in, False, True, MIN_DF)

#TFIDF
#MAX_F = 3500
features = tfidf_vectorization(file_in, max_f = None)


#Google-News Word2Vector Model
'''
modelw2v = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary = True)
MAX_LENGTH = 250
#create a MAX_LENGTH by 300 matrix for each text
dataset = pd.read_excel(file_in)
vectorized_headlines = vectorize_data(dataset["Text"], modelw2v, 300, MAX_LENGTH)
features = np.array(vectorized_headlines)# convert into numpy array
print(features.shape)
features = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
'''
AVG_COUNT = 0

avg_acc = [0,0,0,0,0]
avg_pr = [0,0,0,0,0]
avg_rc = [0,0,0,0,0]
avg_f1 = [0,0,0,0,0]

kf = KFold(n_splits=5, shuffle = False)

for train_indices, test_indices in kf.split(features):
    X_train = features[train_indices]
    X_test =  features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    #X_train, X_test, y_train, y_test = train_test_split(features,labels)
    X_train, y_train = MLSMOTE(X_train, y_train, n_neighbors=5, target_ratio=1.0)
    #LOGISTIC REGRESSION
    clf = MultiOutputClassifier(estimator= LogisticRegression(max_iter = 500)).fit(X_train, y_train)
    #SUPPORT VECTOR CLASSIFIER
    #clf = MultiOutputClassifier(estimator=SVC(kernel='linear', C=1.0)).fit(X_train, y_train)
    
    yhat = clf.predict(X_test)
    

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
            
    conf = [anger, fear, disgust, sadness, joy]
        
    print("RUN")
    print ("Accuracy:\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (accuracy(anger), accuracy(fear), accuracy(disgust),accuracy(sadness), accuracy(joy)))
    print("Precision:\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (precision(anger), precision(fear), precision(disgust), precision(sadness), precision(joy)))
    print("Recall:\t\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" %  (recall(anger), recall(fear), recall(disgust), recall(sadness), recall(joy)))
    try:
    
        print("F1-score:\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" %  (f1(anger), f1(fear), f1(disgust), f1(sadness), f1(joy)))
        print("F1-average:\t%.3f" % ((f1(anger)+ f1(fear)+ f1(disgust)+ f1(sadness)+ f1(joy))/5))
        AVG_COUNT += 1
        for i in range(5):
            avg_acc[i] += accuracy(conf[i])
            avg_rc [i] += recall(conf[i])
            avg_pr [i] += precision(conf[i])
            avg_f1 [i] += f1(conf[i])
        
    except ZeroDivisionError:
        print("F1 score cannot be calculated ")

for i in range(5):
    avg_acc[i] /= AVG_COUNT
    avg_pr[i] /=AVG_COUNT
    avg_rc[i] /=AVG_COUNT
    avg_f1[i] /= AVG_COUNT

print("AVERAGE")
print("Accuracy\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (avg_acc[0], avg_acc[1], avg_acc[2], avg_acc[3], avg_acc[4]))
print("Precision\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (avg_pr[0], avg_pr[1], avg_pr[2], avg_pr[3], avg_pr[4]))
print("Recall\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (avg_rc[0], avg_rc[1], avg_rc[2], avg_rc[3], avg_rc[4]))
print("F1-score\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (avg_f1[0], avg_f1[1], avg_f1[2], avg_f1[3], avg_f1[4]))

print("F1-avg\t%.3f" % (sum(avg_f1)/5))

