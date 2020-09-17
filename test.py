from urllib.request import urlopen, Request
import time
import joblib
import pickle
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import urllib
from sklearn.preprocessing import MaxAbsScaler
from tkinter import messagebox as msg
from tkinter import Tk

def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    tkns_BySlash = f.split("\n")
    for i in range(len(tkns_BySlash)):
        total_Tokens = total_Tokens+re.findall(r"[\w']+", tkns_BySlash[i])
    total_Tokens = list(set(total_Tokens))
    return total_Tokens




vectorizer = pickle.load(open("vectorizer.pickle",'rb'))
model = joblib.load('grad.pkl')


current_url=["kalantzis.net","www.naver.com","zkic.com","princetonbasketball.com/?cat=28&paged=3"]
X_pre = vectorizer.transform(current_url)
New_pre = model.predict(X_pre)

print(New_pre)

        

