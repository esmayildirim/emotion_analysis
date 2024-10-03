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


def model_mlp(X, y):
    model = Sequential()
    model.add(Dense(1024, input_dim = X.shape[1], kernel_initializer= 'he_uniform',bias_initializer='he_uniform', activation = 'relu'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(Dropout(0.35))
    #model.add(Dense(512, kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform', activation ='relu'))
    #model.add(Dropout(0.35))
    #model.add(Dense(512, kernel_initializer= 'he_uniform', bias_initializer= 'he_uniform', activation ='relu'))
    #model.add(Dropout(0.35))
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
    

if __name__ == '__main__':
    file = "lemmatized_TR.xlsx"
    #X = bag_of_words_matrix(file)
    #X = count_vectorizer(file, False, True, 1)
    X = tfidf_vectorization(file, max_f = None)
    #X = word2vec_vectorization(file, 10)
    y = emotion_matrix(file)
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
    print(emotion_count)
            
    
    
    #multiple runs lists
    anger_list = [] # each element is a list of acc, prec, recall
    fear_list = []
    sadness_list = []
    disgust_list = []
    joy_list = []
    RANGE = 1
    for i in range(RANGE):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model_train = model_mlp(X,y)
        capture_cb = WeightCapture(model_train)
        capture_cb.on_epoch_end(-1)
        history = model_train.fit(X_train, y_train, epochs=150,callbacks=[capture_cb], verbose=1, validation_split=0.2)
        #plotweight(capture_cb)
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','val'], loc='upper right')
        plt.show()
        
        yhat = model_train.predict(X_test)
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
        print("RUN ", i)
        print ("Accuracy:\t%.2f %.2f %.2f %.2f %.2f" % (accuracy(anger), accuracy(fear), accuracy(disgust),accuracy(sadness), accuracy(joy)))
        print("Precision:\t%.2f %.2f %.2f %.2f %.2f" % (precision(anger), precision(fear), precision(disgust), precision(sadness), precision(joy)))
        print("Recall:\t\t%.2f %.2f %.2f %.2f %.2f" %  (recall(anger), recall(fear), recall(disgust), recall(sadness), recall(joy)))
        
        anger_list.append([accuracy(anger), precision(anger), recall(anger)])
        fear_list.append([accuracy(fear), precision(fear), recall(fear)])
        sadness_list.append([accuracy(sadness), precision(sadness), recall(sadness)])
        disgust_list.append([accuracy(disgust), precision(disgust), recall(disgust)])
        joy_list.append([accuracy(joy), precision(joy), recall(joy)])
        
    #calculate means
    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    for k in range(RANGE):
        accuracy_mean += anger_list[k][0]
        precision_mean += anger_list[k][1]
        recall_mean += anger_list[k][2]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    
    print("ANGER MEAN RESULTS:", accuracy_mean, precision_mean, recall_mean)
    
    #complete for the other emotions
    
    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    for k in range(RANGE):
        accuracy_mean += fear_list[k][0]
        precision_mean += fear_list[k][1]
        recall_mean += fear_list[k][2]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    
    print("FEAR MEAN RESULTS:", accuracy_mean, precision_mean, recall_mean)
    
    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    for k in range(RANGE):
        accuracy_mean += sadness_list[k][0]
        precision_mean += sadness_list[k][1]
        recall_mean += sadness_list[k][2]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    
    print("SADNESS MEAN RESULTS:", accuracy_mean, precision_mean, recall_mean)


    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    for k in range(RANGE):
        accuracy_mean += disgust_list[k][0]
        precision_mean += disgust_list[k][1]
        recall_mean += disgust_list[k][2]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    
    print("DISGUST MEAN RESULTS:", accuracy_mean, precision_mean, recall_mean)


    accuracy_mean = 0
    precision_mean = 0
    recall_mean = 0
    for k in range(RANGE):
        accuracy_mean += joy_list[k][0]
        precision_mean += joy_list[k][1]
        recall_mean += joy_list[k][2]
    accuracy_mean /= RANGE
    precision_mean /= RANGE
    recall_mean /= RANGE
    
    print("JOY MEAN RESULTS:", accuracy_mean, precision_mean, recall_mean)
