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

#load list of stop words like : "a", "and", "but", "how", "or", and "what"
stopWords = list(stopwords.words("english"))
#Load WordNet dictionary lemmatizer to convert various words to base form example: studies -> study
lemmatizer = WordNetLemmatizer()
#tokenizer splits the string passed on the basis of regular expression provided.
tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)
#regular expressions for various parts of speech (POS)
noun=re.compile("^NNP$")
pronoun=re.compile(r"^W.+|^P.+")

def tokenize_text(text):
    #convert text to lower case
    text = text.lower()
    #split text into tokens
    tokens = tokenizer.tokenize(text)
    #Adds tags to tokens as per POS tags, needs tokenized text as input
    tagged_tokens=pos_tag(tokens)
    #array to store processed tokens
    tokens=[]
    #If token has tag of noun or pronoun then it should be ignored other wise considered
    for token in tagged_tokens:
        if(not pronoun.match(token[1]) and not noun.match(token[1])):
            tokens.append(token[0])

    lemmatized_tokens = []
    #Use lemmatizer to convert tokens into base form
    for token in tokens:
        token = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(token)
    token = lemmatized_tokens

    #remove stopWords from tokens and remove tokens with 1 length as they are insignificant
    updated_tokens = []
    for token in tokens:
        if(token not in stopWords and len(token) > 1):
            updated_tokens.append(token)

    #return updated tokens
    return updated_tokens
