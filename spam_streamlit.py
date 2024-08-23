import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.title("Spam Email Detection")
df=pd.read_csv('spam.csv')
df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)

#split data to trainin and testing  data 
x_train,x_test,y_train,y_test=train_test_split(df.Message,df.spam,test_size=0.25)

#create pipeline to count vectorize text and perform mutlinomialnb
pipe=Pipeline([
    ('cv',CountVectorizer()),
    ('mt',MultinomialNB())
])
pipe.fit(x_train,y_train)
st.info("## Dataset")
st.write(df)
st.info("## Model Accuarcy")
st.write(pipe.score(x_test,y_test)*100)

str=st.text_input("Enter your Email message  to detect whether it is spam or ham")
str=[str]
if st.button('Detect'):
    out=pipe.predict(str)
    if out==0:
        st.success("Ham")
    elif out==1:
        st.success("Spam")