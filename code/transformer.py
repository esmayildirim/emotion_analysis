import keras 
from keras import ops
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from transformer_data import load_data # my library
from mlsmote_lib import MLSMOTE # my library
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

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

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

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


def prep_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim):
    
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(5, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    vocab_size = 3000  # Only consider the top 20k words # TRY 2000, 3000, 4000
    maxlen = 100  # Only consider the first 200 words of each text TRY 100, 200, 300
    embed_dim = 32  # Embedding size for each token TRY 16, 32, 64
    num_heads = 2  # Number of attention heads TRY 2, 4
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer TRY 16, 32, 64
    
    #TOTAL TESTS (3 x 3 x 3 x 2 x 3 x 4 = 648 TESTS)


    # words in this dataset have been encoded with integer values
    # 3 means 3rd most frequent word
    file_path = "../stemmed_ENG.xlsx"
    X = load_data(file_path, vocab_size, maxlen)
    X = keras.utils.pad_sequences(X, maxlen = maxlen)
    X = np.array(X)
    y = emotion_matrix(file_path)

    avg_acc = [0,0,0,0,0]
    avg_pr = [0,0,0,0,0]
    avg_rc = [0,0,0,0,0]
    avg_f1 = [0,0,0,0,0]
    kf = KFold(n_splits=5, shuffle = False)
    AVG_COUNT = 0

    for train_indices, test_indices in kf.split(X):
        #X_train, X_test, y_train, y_test = train_test_split(X,y)
        X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]


        #X_train, y_train = MLROS(X_train, y_train, target_size = 500)
        X_train, y_train = MLSMOTE(X_train, y_train, n_neighbors=5, target_ratio=1.0)
        print("Resampled label counts:", y_train.sum(axis=0))
        print("X_res shape:", X_train.shape)
        
        #TRAIN
        model = prep_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim)
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

        checkpoint = ModelCheckpoint('transformer_best_weights.keras',
                                     monitor='val_loss',
                                     verbose=0,
                                     mode='min',
                                     save_best_only=True)

        history = model.fit(
            X_train, y_train, batch_size=32, callbacks = [checkpoint], epochs=250, verbose = 0, validation_split = 0.2)

        #plt.plot(history.history['accuracy'])
        #plt.plot(history.history['val_accuracy'])
        #plt.title('accuracy')
        #plt.ylabel('accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train','val'], loc='upper right')
        #plt.show()
        #TEST
        model.load_weights('transformer_best_weights.keras')
        yhat = model.predict(X_test)
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
        
        conf = [anger, fear, disgust, sadness, joy]
        
        
        print("FOLD ")
        print ("Accuracy:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (accuracy(anger), accuracy(fear), accuracy(disgust),accuracy(sadness), accuracy(joy)))
        print("Precision:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (precision(anger), precision(fear), precision(disgust), precision(sadness), precision(joy)))
        print("Recall:\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %  (recall(anger), recall(fear), recall(disgust), recall(sadness), recall(joy)))
        
        try:
        
            print("F1-score:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %  (f1(anger), f1(fear), f1(disgust), f1(sadness), f1(joy)))
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

    print("AVG RESULTS: vocab_size = ", vocab_size, ", maxlen = ", maxlen, ", embed_dim = ", embed_dim,
             ", num_heads = ", num_heads, ", ff_dim = ", ff_dim)
    print("Accuracy\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (avg_acc[0], avg_acc[1], avg_acc[2], avg_acc[3], avg_acc[4]))
    print("Precision\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (avg_pr[0], avg_pr[1], avg_pr[2], avg_pr[3], avg_pr[4]))
    print("Recall\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (avg_rc[0], avg_rc[1], avg_rc[2], avg_rc[3], avg_rc[4]))
    print("F1-score\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (avg_f1[0], avg_f1[1], avg_f1[2], avg_f1[3], avg_f1[4]))

    print("F1-avg\t%.3f" % (sum(avg_f1)/5))


