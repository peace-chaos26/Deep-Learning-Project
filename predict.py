# Adding necessary imports
import csv
import re
import nltk
from nltk import pos_tag
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn import svm
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.stem import PorterStemmer,SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import LogisticRegression as LR
from nltk.stem import WordNetLemmatizer
import pickle
from PyLyrics import *
import common as common
import os
import eyed3

# Path to read available songs
path = "C:\Users\sakshi\Desktop\Minor\Minor-master\songs"
# Path to store songs based on sentiment

updatedPath = "C:\Users\sakshi\Desktop\Minor\Minor-master\songs"
#read all files at path
audioFiles = os.listdir(path)
print(audioFiles)
#load saved vectorizer and model during training
vectorizer = pickle.load(open('dictionary.pkl','rb'))
model=pickle.load(open('finalized_model.sav','rb'))
emotionDictionary = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "relaxed"
}
def sort_songs():
    for audioFile in audioFiles:
        #Excract information from audioFile
        try:
            audio = eyed3.load(path+audioFile)
            singer = audio.tag.artist
            song = audio.tag.title
            print("Processing song...")
            print("Song title: ", song)
            print("Song Artist: ", singer)
            # Fetch lyrics using pylyrics
            lyrics = PyLyrics.getLyrics(singer,song)
            print("Lyrics: ")
            print(lyrics)
            #Convert to array
            lyrics = [lyrics]
            #Vectorize the lyrics
            X_input = vectorizer.transform(lyrics)
            # Using trained model, get sentiment of song
            print(type(model.predict(X_input)))
            sentiment = emotionDictionary[list(model.predict(X_input))[0]]
            print("Sentiment: "+sentiment)
            #move the song to new directory
            p = updatedPath+sentiment+"/"+audioFile
            print(p)
            os.rename(path+audioFile, p)
        except:
            print("Can't analyse "+audioFile)

def getSentiment(model, input):
    return model.predict(input)

sort_songs()
