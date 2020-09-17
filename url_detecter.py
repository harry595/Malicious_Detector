from selenium import webdriver
from bs4 import BeautifulSoup 
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

# parsing part 
def make_link_list(url):
    Dict={}
    req=urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html=urllib.request.urlopen(req)
    tmp=BeautifulSoup(html,"html.parser")

    for link in tmp.find_all('a'):
        tmp=link.get('href')
        value=link.text.strip()[0:10]
        if(len(value)>=10):  #열글자 넘을 경우
            value+="..."
        try:
            if('http' in tmp):   
                Dict[value]=tmp  #Dict[tmp]=value
        except:
            continue
    return Dict.values()



def makeTokens(f):
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    tkns_BySlash = f.split("\n")
    for i in range(len(tkns_BySlash)):
        total_Tokens = total_Tokens+re.findall(r"[\w']+", tkns_BySlash[i])
    total_Tokens = list(set(total_Tokens))
    return total_Tokens



#train, test data parameter
TEST_SIZE = 0.25
RS = 42
#Open data
urls_data = pd.read_csv("data.csv")
type(urls_data)
urls_data.head()

#Separate URL and label
y = urls_data["label"]
url_list = urls_data["url"]
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
X = vectorizer.fit_transform(url_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state = RS)

current_url="data:,"
vectorizer = pickle.load(open("vectorizer.pickle",'rb'))
model = joblib.load('grad.pkl')
driver = webdriver.Chrome(executable_path=r'./chromedriver.exe')

root=Tk()
root.withdraw()

while 1:
    time.sleep(1)
    #Req_url = Request(driver.current_url, headers={'User-Agent':'Mozilla/5.0'})

    if(current_url != driver.current_url):
        current_url=driver.current_url
        url_list = make_link_list(current_url)
        X_pre = vectorizer.transform(url_list)
        New_pre = model.predict(X_pre)
        print(New_pre)
        if 'bad' in New_pre:
            msg.showwarning('Warning : malicious Url Detected.', 'becareful, don\'t click link' ) 

