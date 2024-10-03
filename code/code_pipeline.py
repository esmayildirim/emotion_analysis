#Code Pipeline for Sentiment Analysis

import pandas as pd
# import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from zemberek import(
       #TurkishSpellChecker,
       TurkishSentenceNormalizer,
       TurkishSentenceExtractor,
       TurkishMorphology,
       #TurkishTokenizer,
       )
import time
import logging
from textblob import TextBlob
from textblob.exceptions import NotTranslated
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import re  
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from gensim.models import Word2Vec


def normalization(filein, fileout):  #Note - check for Nan Values
   extractor = TurkishSentenceExtractor()
   morphology = TurkishMorphology.create_with_defaults()
   normalizer = TurkishSentenceNormalizer(morphology)
   dataset = pd.read_excel(filein)
   Dictionary = dataset.to_dict('list')
   texts = dataset["Text"]

   new_texts = [] #normalization process below:
   for paragraphs in texts:
       sentences = extractor.from_paragraph(paragraphs) #extract all the sentences from the paragraph
       new_paragraphs = ""
       for i in sentences:
           normalized_sentence = normalizer.normalize(i)
           new_paragraphs += normalized_sentence + " "
       new_texts.append(new_paragraphs)
    
   Dictionary.update({'Text': new_texts})
   df = pd.DataFrame(Dictionary)
   df.to_excel(fileout, index = False)
   print(dataset['Text'], "\n", df['Text'])
   print("Finished") #Takes around 3 minutes to run


def translate(filein, fileout): #translatr from Turkish to English
   # Step 1: Translate the text column
   dataset = pd.read_excel(filein)
   Dictionary = dataset.to_dict('list')
   texts = dataset["Text"]
   nan_indices = texts.isna() #finds nan value indices as True/False
   new_texts = []
   i = 0
   for paragraph in texts:
       if(nan_indices[i] == False):
        #    paragraph = str(paragraph)
           blob = TextBlob(paragraph)
           paragraph = blob.translate(from_lang='tr', to='en')
       new_texts.append(paragraph)
       i += 1
  
   # Step 2: Translate the title column
   new_titles = []
   titles = dataset[str('Title')] #Nan value, numbers & date in title
   count = 1
   for i in titles:
       blob = TextBlob(str(i))
       try: #Exception handling for Dates since they can't be translated
           new_title = blob.translate(from_lang='tr', to='en')
       except NotTranslated:
           print("Could not translate ", i)
       new_titles.append(str(new_title))
       count += 1

   # Step 3: Translate the genre column
   genres = dataset[str("Genre")]
   new_genres = []
   for i in genres:
       blob = TextBlob(str(i))
       try:
           new_genre = blob.translate(from_lang = 'tr', to='en')
       except NotTranslated:
           print("Could not translate ", i)
       new_genres.append(str(new_genre))

   # Step 4: Save translated Data into new excel file
   Dictionary.update({'Text': new_texts,'Title': new_titles, 'Genre': new_genres})
   df = pd.DataFrame(Dictionary)
   df.to_excel(fileout, index = False)
   print(dataset, "\n", df)
   print("Finished")



def english_stopword_removal_lemmatization_stemming(filein, fileout_stem, fileout_lemma):
   df = pd.read_excel(filein)
   Dictionary = df.to_dict('list')
   corpus = pd.Series(df.Text.tolist()).astype(str)
   
   def text_clean(corpus, keep_list):
       cleaned_corpus = pd.Series()# empty vector
       for row in corpus:
           qs = []
           for word in row.split():
               if word not in keep_list:
                   p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                   p1 = p1.lower()
                   qs.append(p1)
               else : qs.append(word)
        #    cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs))) # adds the cleaned sentence back to the empty vector
           cleaned_corpus = cleaned_corpus.tolist()
           cleaned_corpus.append(' '.join(qs))
           cleaned_corpus = pd.Series(cleaned_corpus)
       return cleaned_corpus

   def stopwords_removal(corpus):
       wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
       stop = set(stopwords.words('english'))
       for word in wh_words:
           stop.remove(word)
       corpus = [[x for x in x.split() if x not in stop] for x in corpus]
       return corpus

   def lemmatize(corpus):
       lem = WordNetLemmatizer()
       corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
       return corpus

   def stem(corpus, stem_type = None):
       if stem_type == 'snowball':
           stemmer = SnowballStemmer(language = 'english')
           corpus = [[stemmer.stem(x) for x in x] for x in corpus]
       else :
           stemmer = PorterStemmer()
           corpus = [[stemmer.stem(x) for x in x] for x in corpus]
       return corpus

   def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):
       if cleaning == True:
           corpus = text_clean(corpus, keep_list)

       if remove_stopwords == True:
           corpus = stopwords_removal(corpus)
       else :
           corpus = [[x for x in x.split()] for x in corpus]
      
       if lemmatization == True:
           corpus = lemmatize(corpus)
      
       if stemming == True:
           corpus = stem(corpus, stem_type)
      
       corpus = [' '.join(x) for x in corpus]       
       return corpus
   common_dot_words = ['U.S.A', 'Mr.', 'Mrs.', 'D.C.']
    
   # Main Functions to be called below
   corpus_with_lemmatization = preprocess(corpus, keep_list = common_dot_words, stemming = False, stem_type = None, lemmatization = True, remove_stopwords = True)
   corpus_with_stemming = preprocess(corpus, keep_list = common_dot_words, stemming = True, stem_type = "snowball", lemmatization = False, remove_stopwords = True)
   
   #Lemmatization Procedure - make an empty list for lemmatization
   lemma_list = []
   for i in range(len(df['Title'])): #417 rows right now, but might change so wrote len(df['Title']) aka how many rows there are
       lemma_list.append(corpus_with_lemmatization[i])
        # --> Put Lemmatized data into an excel file
   Dictionary.update({'Text': lemma_list})
   dfLemma = pd.DataFrame(Dictionary)
   dfLemma.to_excel(fileout_lemma, index = False)

   #Stemming Procedure - make an empty list for stemming:
   stem_list = []
   for i in range(len(df['Title'])):
       stem_list.append(corpus_with_stemming[i])
        # --> Store Lemmatized Data in an excel file       
   Dictionary.update({'Text': stem_list})
   dfStemming = pd.DataFrame(Dictionary)
   dfStemming.to_excel(fileout_stem, index = False)
   print("Finished") #Finished quickly 






#Removing Turkish Stopwords, Lemmatizing, Stemming Data | Creating two excel files(Stem and Lemma)
def turkish_stopword_removal_lemmatization_stemming(filein, fileout_stem, fileout_lemma):
    logger = logging.getLogger(__name__)
    morphology = TurkishMorphology.create_with_defaults()
    start = time.time()
    extractor = TurkishSentenceExtractor()
    print("Extractor instance created in: ", time.time() - start, "s")
    zemberek_tr_stop = []
    with open('stop-words.tr.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            zemberek_tr_stop.append(line.strip())
    #print(zemberek_tr_stop)

    dataset = pd.read_excel(filein)
    Dictionary = dataset.to_dict('list')
    Texts = dataset["Text"]

    StemmedParagraphList = []
    LemmaParagraphList = []

    for paragraph in Texts:
        sentences = extractor.from_paragraph(paragraph)
        stemmedParagraph = ""
        lemmaParagraph = ""

        for sentence in sentences:
            stemmedSentence = ""
            lemmaSentence = ""
            analysis = morphology.analyze_sentence(sentence)
            after = morphology.disambiguate(sentence, analysis)

            for word in after.best_analysis():
                if(word.item.primary_pos.short_form != "Punc" and word.item.primary_pos.short_form != "Unk" and word.item.lemma not in zemberek_tr_stop):
                    stemmedSentence += word.get_stem() + " "
                    lemmaSentence += word.item.lemma + " "

            stemmedParagraph += stemmedSentence
            lemmaParagraph += lemmaSentence

        StemmedParagraphList.append(stemmedParagraph)
        LemmaParagraphList.append(lemmaParagraph)

    # Creating 2 Excel Files for Stemmed and Lemmatized Turkish data
    Dictionary.update({'Text': StemmedParagraphList})   
    Stemdf = pd.DataFrame(Dictionary)
    Stemdf.to_excel(fileout_stem, index = False)

    Dictionary.update({'Text': LemmaParagraphList})
    Lemmadf = pd.DataFrame(Dictionary)
    Lemmadf.to_excel(fileout_lemma, index = False)
    print(Stemdf['Text'], Lemmadf['Text'])
    print(time.time())




def bag_of_words_matrix(file):
   dataset = pd.read_excel(file)
   Text = dataset["Text"] # Texts that don't have periods/punctuation so they're just sentences
   print("Length of Texts:", len(Text))
  
   # Build Dictionary
   set_of_words = set()
   for sentence in Text:
       for word in sentence.split():
           set_of_words.add(word)
   vocab = list(set_of_words)
   print("vocabulary length: ", len(vocab), "number of words")

   # Get position of each word in the matrix
   position = {}
   for i, token in enumerate(vocab):
       position[token] = i

   #Create an empty matrix with ~430 rows and (# of words in the vocab) columns
   bow_matrix = np.zeros((len(Text), len(vocab)))

   # Fill in matrix. Number represents frequency of word in text
   for i, sentence in enumerate(Text):
       for word in sentence.split():
           bow_matrix[i][position[word]] +=  1 # bow_matrix[i][position[word]] = bow_matrix[i][position[word]] + 1
  
   # View Matrix
   print("Length of Matrix: ", len(bow_matrix))
   print("Shape of Matrix: ", np.shape(bow_matrix))
   df = pd.DataFrame(data = bow_matrix.astype(float))
   print(df)
   return(df)
   

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

        



def count_vectorizer(file, set_ngrams=False, min_df=False, min_df_value = 1, max_df=False, max_df_value=50):
   dataset = pd.read_excel(file)
   paragraphs = dataset["Text"]
   vectorizer = CountVectorizer(analyzer='word')
   if min_df == True:
       vectorizer.set_params(min_df=min_df_value)
   if max_df == True:
       vectorizer.set_params(max_df=max_df_value)
   if set_ngrams == True:
       vectorizer.set_params(ngram_range = (1,3))
     # Save matrix
   bow_matrix = vectorizer.fit_transform(paragraphs)
   df = pd.DataFrame(data=bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
   
   # Print # of rows & columns
   print(vectorizer.get_feature_names_out())
   print(bow_matrix.toarray())
   num_documents, num_features = bow_matrix.shape
   print(f"Number of documents (rows): {num_documents}")
   print(f"Number of features (columns): {num_features}")
   return bow_matrix.toarray()


#This function below will be called by the three other functions below this function to create 12 graphs, related to Count Vectorizer & English/Turkish Stem/Lemma
def Count_Vectorizer_Graph(file, set_max_df = False, max_df_value = 25, set_min_df = False, min_df_value = 5):
    Dataset = pd.read_excel(file)
    Paragraphs = Dataset["Text"]
    vectorizer = CountVectorizer(analyzer='word')
    
    #Responding to user input
    if set_max_df == True:
       vectorizer.set_params(max_df=max_df_value)

    if set_min_df == True:
       vectorizer.set_params(min_df=min_df_value)
    
    bow_matrix_min_features = vectorizer.fit_transform(Paragraphs)
    num_documents, num_features = bow_matrix_min_features.shape
    #print(f"Minimum Threshold: {min_df_value}, Number of words in vocabulary: {num_features}")
    return num_features

def Make_Count_Vectorizer_Min_Graphs(file):
    Value5 = Count_Vectorizer_Graph(file, set_min_df = True, min_df_value=5)
    Value10 = Count_Vectorizer_Graph(file, set_min_df = True, min_df_value=10)
    Value15 = Count_Vectorizer_Graph(file, set_min_df = True, min_df_value=15)
    Value20 = Count_Vectorizer_Graph(file, set_min_df = True, min_df_value=20)
    Value25 = Count_Vectorizer_Graph(file, set_min_df = True, min_df_value=25)
    #put numbers into dataframe and graph it:
    Dictionary = {'Minimum Threshold': [5, 10, 15, 20, 25], 'Number of Words in the Vocabulary': [Value5, Value10, Value15, Value20, Value25]}
    df = pd.DataFrame(Dictionary)
    print(df)
    filename = file.split(".")[1]
    df.plot(x='Minimum Threshold', y='Number of Words in the Vocabulary' , kind='scatter', title=f'{filename} Min Threshold Matrices')	    
    plt.show() #If have more time, show the exact (x,y) values by the points

def Make_Count_Vectorizer_Max_Graphs(file):
   Value5 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=5)
   Value10 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=10)
   Value15 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=15)
   Value20 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=20)
   Value25 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=25)
    #put numbers into dataframe and graph it:
   Dictionary = {'Maximum Threshold': [5, 10, 15, 20, 25], 'Number of Words in the Vocabulary': [Value5, Value10, Value15, Value20, Value25]}
   df = pd.DataFrame(Dictionary)
   print(df)
   filename = file.split(".")[1]
   df.plot(x='Maximum Threshold', y='Number of Words in the Vocabulary' , kind='scatter', title=f' {filename} Max Threshold Matrices')	    
   plt.show() #If have more time, show the exact (x,y) values by the points

def Make_Count_Vectorizer_Min_and_Max_Graphs(file):
   #Value_min_and_max format
   Value5and10 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=10, set_min_df = True, min_df_value=5)
   Value5and15 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=15, set_min_df=True, min_df_value=5)
   Value5and20 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=20, set_min_df=True, min_df_value=5)
   Value5and25 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=25, set_min_df=True, min_df_value=5)
   Value10and15 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=15, set_min_df=True, min_df_value=10)
   Value10and20 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=20, set_min_df=True, min_df_value=10)
   Value10and25 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=25, set_min_df= True, min_df_value=10)
   Value15and20 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=20, set_min_df=True, min_df_value=15)
   Value15and25 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=25, set_min_df=True, min_df_value=15)
   Value20and25 = Count_Vectorizer_Graph(file, set_max_df = True, max_df_value=25, set_min_df=True, min_df_value=20)
        # --> put numbers into dataframe and graph it:
   Dictionary = {'Minimum & Maximum Thresholds': ['5 & 10', '5 & 15', '5 & 20', '5 & 25', '10 & 15', '10 & 20', '10 & 25', '15 & 20', '15 & 25', '20 & 25'], 'Number of Words in the Vocabulary': [Value5and10, Value5and15, Value5and20, Value5and25, Value10and15, Value10and20, Value10and25, Value15and20, Value15and25, Value20and25]}
   df = pd.DataFrame(Dictionary)
   print(df)
   filename = file.split(".")[1] # What if the file name isn't "./filename.xlsx" but "filename.xlsx", make a decision later to keep/remove this
   df.plot(x='Minimum & Maximum Thresholds', y='Number of Words in the Vocabulary', kind='scatter', title=f'{filename} Max & Min Threshold Matrices')	    
   plt.show() #If have more time, show the exact (x,y) values by the points



def tfidf_vectorization(file, normp='l1', analyzer='word', ngram_r= (1,3), max_f= None):
   # Read dataset
   dataset = pd.read_excel(file)
   texts = dataset["Text"]

   # Just in case there are NaN values
   texts = [text if pd.notna(text) else "" for text in texts]

   # Initialize TfidfVectorizer
   if max_f != None:
        vectorizer = TfidfVectorizer(max_features = max_f)
   else:
        vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform(texts)

   # Print tf-idf matrix
   print("Feature names:", vectorizer.get_feature_names_out())
   print("TF-IDF matrix:\n", tfidf_matrix.toarray())
   print("Shape of matrix:", tfidf_matrix.shape)
   
   #vectorizer_ngram_max_features = TfidfVectorizer(norm = normp, analyzer = 'word', ngram_range = ngram_r, max_features = tfidf_matrix.shape[0])
   #tf_idf_matrix_n_gram_max_features = vectorizer_ngram_max_features.fit_transform(texts)
    #print(vectorizer_ngram_max_features.get_feature_names.out())
   #print("The shape of the ngram",ngram_r, " max_features=",max_f , "matrix is ", tf_idf_matrix_n_gram_max_features.shape, "\n End of tfidf_vectorization function \n")
   return tfidf_matrix.toarray()




def word2vec_vectorization(file_name, VECTOR_SIZE = 100):
   # Read dataset
   Dataset = pd.read_excel(file_name)
   X_train = Dataset["Text"]

   # Tokenize sentences
   sentences = [sentence.split() for sentence in X_train]

   # Train Word2Vec model
   w2v_model = Word2Vec(sentences, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4)

   # Define vectorization function
   def vectorize(sentence):
       words = sentence.split()
       words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]

       if len(words_vecs) == 0:
           return np.zeros(VECTOR_SIZE)
       words_vecs = np.array(words_vecs)
       print("words_vecs shape", words_vecs.shape)
       return words_vecs.mean(axis=0)

   # Apply vectorization function to X_train
   X_train_vectorized = np.array([vectorize(sentence) for sentence in X_train])
   print("Training shape:", X_train_vectorized.shape)
   return X_train_vectorized


    # Making graphs on normalized english dataset, mostly metadata
def Documents_by_Gender(file):    #Number of Documents vs Gender (Male, Female, Other)
    dataset = pd.read_excel(file)
    Gender = dataset["Gender"]
    
    FemaleList = []
    MaleList = []
    OtherList = []
    for i in Gender:
        if i == "F":
            FemaleList.append(i)
        
        elif i == "M":
            MaleList.append(i)

        else:
            OtherList.append(i)

    female_count = len(FemaleList) # 315 documents by female gender
    male_count = len(MaleList) # 59 documents by male gender
    other_count = len(OtherList) # 42 documents w/ blank gender

    Dictionary = {"Female": [female_count], "Male": [male_count], "Other": [other_count]}
    df = pd.DataFrame(Dictionary)
    print(df)
    df.plot(kind='bar', title="Number of Documents by Gender", xlabel='Gender', ylabel='Number of Documents', figsize=(7, 5))
    plt.legend(loc="upper right")
    plt.grid(axis='y')
    plt.show()

def Average_Words_By_Gender(file):
    dataset = pd.read_excel(file)
    Text = dataset["Text"]    
    Gender = dataset["Gender"]

    female_text = []
    male_text = []
    other_text = []
    female_word_count_list = []
    male_word_count_list = []
    other_word_count_list = []

    for i in range(len(Text)):
        if Gender[i] == "F":
            female_text.append(Text[i])
            word_count = 0
            sentences = Text[i].split(".")
            for j in range(len(sentences)):
                word = sentences[j].split(" ")
                word_count += len(word)
            female_word_count_list.append(word_count)

        elif Gender[i] == "M":
            male_text.append(Text[i])
            word_count = 0
            sentences = Text[i].split(".")
            for j in range(len(sentences)):
                word = sentences[j].split(" ")
                word_count += len(word)
            male_word_count_list.append(word_count)

        else:
            other_text.append(Text[i])
            word_count = 0
            sentences = Text[i].split(".")
            for j in range(len(sentences)):
                word = sentences[j].split(" ")
                word_count += len(word)
            other_word_count_list.append(word_count)

    female_average_word_count = sum(female_word_count_list) / len(female_word_count_list)
    male_average_word_count = sum(other_word_count_list) / len(male_word_count_list)
    other_average_word_count = sum(other_word_count_list) / len(other_word_count_list)

    print(female_average_word_count)
    print(male_average_word_count)
    print(other_average_word_count)

    #Graph data
    Dictionary = {"Female_Average_Word": round(female_average_word_count), "Male_Average_Word": round(male_average_word_count), "Other_Average_Word": round(other_average_word_count)}
    df = pd.DataFrame(Dictionary, index=[0])
    print(df)
    df.plot(kind='bar', title="Average Words in Each Text Sorted by Gender", xlabel='Gender', ylabel='Average Number of Words', figsize=(12, 7))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

    dataset = pd.read_excel(file)
    Gender = dataset["Gender"]
    
    FemaleList = []
    MaleList = []
    OtherList = []
    for i in Gender:
        if i == "F":
            FemaleList.append(i)
        
        elif i == "M":
            MaleList.append(i)

        else:
            OtherList.append(i)

    female_count = len(FemaleList) # 315 documents by female gender
    male_count = len(MaleList) # 59 documents by male gender
    other_count = len(OtherList) # 42 documents w/ blank gender

    Dictionary = {"Female": [female_count], "Male": [male_count], "Other": [other_count]}
    df = pd.DataFrame(Dictionary)
    print(df)
    df.plot(kind='bar', title="Number of Documents by Gender", xlabel='Gender', ylabel='Number of Documents', figsize=(7, 5))
    plt.legend(loc="upper right")
    plt.grid(axis='y')
    plt.show()

def Documents_by_User(file):
    dataset = pd.read_excel(file)
    Username = dataset["Username"]
    counter_dictionary = Counter(Username)
    df = pd.DataFrame(counter_dictionary, index=[1])
    print(df)
    plt.figure(figsize=(14, 7))
    plt.bar(counter_dictionary.keys(), counter_dictionary.values())
    plt.title("Number of Texts Written by Each User")
    plt.xlabel("Username")
    plt.ylabel("Number of Texts Written")
    plt.grid(axis='y')
    plt.show()
    plt.savefig("Documents_by_User.png") #Error, saved image is blank, but plt.show is good
    



if __name__ == '__main__':
    #normalization("./dataset_labeled_OR.xlsx", "./normalized_TR.xlsx")
    #translate("./normalized_TR.xlsx", "./normalized_ENG.xlsx")
    #english_stopword_removal_lemmatization_stemming("./normalized_ENG.xlsx", "./stemmed_ENG.xlsx", "./lemmatized_ENG.xlsx")
    #turkish_stopword_removal_lemmatization_stemming("./normalized_TR.xlsx", "./stemmed_TR.xlsx", "./lemmatized_TR.xlsx")
   
    # bag_of_words_matrix("./stemmed_TR.xlsx") #Change csv file name in code
    # bag_of_words_matrix("./lemmatized_TR.xlsx") #Change csv file name in code
    # bag_of_words_matrix("./lemmatized_ENG.xlsx") #Change csv file name in code
    # bag_of_words_matrix("./stemmed_ENG.xlsx") #Change csv file name in code

    # emotion_bag_of_word_matrix("./Turkish_Stemmed_Dataset_Labeled.xlsx")
    
    # Count_Vectorizer("./stemmed_TR.xlsx") # If downloading matrix, change code file name
    # Count_Vectorizer("./lemmatized_TR.xlsx")
    # Count_Vectorizer("./lemmatized_ENG.xlsx")
    # Count_Vectorizer("./stemmed_ENG.xlsx")
    
    #Running the functions to make the 12 graphs that are stored in the Google Drive Folder -> Bag of Words -> Graphs
    # Make_Count_Vectorizer_Min_Graphs("./lemmatized_ENG.xlsx")
    # Make_Count_Vectorizer_Min_Graphs("./stemmed_ENG.xlsx")
    # Make_Count_Vectorizer_Min_Graphs("./lemmatized_TR.xlsx")
    #Make_Count_Vectorizer_Min_Graphs("./stemmed_TR.xlsx")

    # Make_Count_Vectorizer_Max_Graphs("./stemmed_TR.xlsx")
    # Make_Count_Vectorizer_Max_Graphs("./lemmatized_ENG.xlsx")
    # Make_Count_Vectorizer_Max_Graphs("./stemmed_ENG.xlsx")
    #Make_Count_Vectorizer_Max_Graphs("./lemmatized_TR.xlsx")

    # Make_Count_Vectorizer_Min_and_Max_Graphs("./stemmed_TR.xlsx")
    # Make_Count_Vectorizer_Min_and_Max_Graphs("./lemmatized_ENG.xlsx")
    # Make_Count_Vectorizer_Min_and_Max_Graphs("./stemmed_ENG.xlsx")
    # Make_Count_Vectorizer_Min_and_Max_Graphs("./lemmatized_TR.xlsx")
   #End of the graphing functions
    
    
    #tfidf_vectorization("./stemmed_TR.xlsx")
    # tfidf_vectorization("./lemmatized_TR.xlsx")
    # tfidf_vectorization("./lemmatized_ENG.xlsx")
    # tfidf_vectorization("./stemmed_ENG.xlsx")

    # word2vec_vectorization("./stemmed_TR.xlsx")
    # word2vec_vectorization("./lemmatized_TR.xlsx")
    # word2vec_vectorization("./lemmatized_ENG.xlsx")
    # word2vec_vectorization("./Stem_English_CRSP_Labeled.xlsx")

    Documents_by_Genre("./normalized_turkish_to_english_labeled.xlsx")
    genre_list("./normalized_turkish_to_english_labeled.xlsx")
    pass
