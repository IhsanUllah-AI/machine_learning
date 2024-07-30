import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.sidebar.title("Insurance Prediction app")
st.sidebar.write("## Additional Options")
#import and dispaly dataframe 
df=pd.read_csv("insurance_data.csv")
st.title("Insurance Classification Based on Age")
st.info("Insurance  Dataset sample ")
st.write(df.head(10))


#visualize data 
st.info("Visualization of Insurance Data set")
fig,ax=plt.subplots()
ax.scatter(df['age'],df['bought_insurance'],marker="+", color='blue')
plt.xlabel("Age")
plt.ylabel("bought_insurance")
plt.title("Insurance system")
st.pyplot(fig)

#load model
import joblib 
model=joblib.load("logisticRegression_model")

#take input  from user 

inp=st.number_input("Enter age to find insurance for it",step=1)
yes="YES"
no="NO"

if st.button("classify"):
    if inp<0:
        st.warning("Age cannot be negative ")
        st.info("please enter correct Age")
    else:
        pre=model.predict(np.array([[inp]]))
        pre_prob=model.predict_proba(np.array([[inp]]))
        #pre_prob has two prob first for no and second for yes 
        #find yes probability and fix it 
        yes_percent=pre_prob[:,1]
        yes_percent=np.fix(yes_percent*100)
        
        #find No probabilty and fix it
        no_perecnt=pre_prob[:,0]
        no_perecnt=np.fix(no_perecnt*100)
        if pre==0:
            st.write("Result")
            st.success(f"bought insurance : {no}")
            st.write("Probablity : ",no_perecnt)  
        else:
            st.write("Result")
            st.success(f"bought insurance : {yes}")
            st.write("Probablity : ",yes_percent)
 

#sidebar 
st.sidebar.info("this app predict whether a user is eligible or not for insurance base on provided age  ")
if st.sidebar.checkbox("Show Raw Data"):
    st.write("Raw Data :",df)

st.sidebar.write("### About ")
st.sidebar.success("this app used logistic regression model to preidct given user is elgible or onot for insurance based on age")
st.sidebar.markdown("[Learrn More ](https://en.wikipedia.org./wiki/Logistic_regression)")