import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


st.sidebar.title("Salary Classification")
st.sidebar.info(""" The decision tree model aims to classify whether a 
        salary exceeds $100k based on company, degree, and job position. By evaluating these factors, 
        the model can provide insights into how various elements influence salary outcomes. 
        Each node in the tree represents a decision point, using historical data to predict 
        salary brackets effectively. This approach simplifies
         complex decision-making processes, making it easier to understand the key drivers of high salaries.
""")

st.title("Decision Tree model to classifiy salary has more or less 100k")
import joblib
model=joblib.load("dec_tree_model")


st.info("Data Set Sample")
df=pd.read_csv("salary_tree.csv")
st.write(df)



#company labeling 
st.write ("**Company Name**")
st.info("Facebook , Google , Amazon")
comp=st.text_input("Enter company name  ").lower()
if comp=="google":
    inp=2
elif comp=="facebook":
    inp=1
elif comp=="amazon":
    inp=0
else :
    st .write("Wrong name enter correct one")


#job libeling
st.info( "Sale_exective  ,  Computer Programer  ,  Buisness_manager")
position=st.text_input("Enter job position").lower()
if position=="sale_exective":
      pos=2
elif position=="computer programer":
    pos=1
elif position=="buisness_manager":
    pos=0
else:
    st.write("wrong position enter correct one")

#degree labeling
st.info("Master  ,  Bachelor")
deg=st.text_input("Enter emply degree master or bachelor").lower()
if deg=="master":
    deg_n=1
elif deg=="bachelor":
    deg_n=0
else:
    st.write("wrong enter correct one")


#Prediction 
if st.button("classify"):
    model_out=model.predict(np.array([[inp,pos,deg_n]]))
    if model_out==1:
        st.write("salary has more 100k $")
        st.success("YES")
    else:
        st.write("salary has more 100k $")
        st.success("NO")
    
if st.sidebar.checkbox("data visulaiztion"):
    inputs=df.drop("salary_more_100k",axis="columns")
    target=df["salary_more_100k"]


    from sklearn.preprocessing import LabelEncoder
    le_company=LabelEncoder()
    le_job=LabelEncoder()
    le_degree=LabelEncoder()


    inputs["company_n"]=le_company.fit_transform(inputs["company"])
    inputs['job_n']=le_job.fit_transform(inputs['job'])
    inputs['degree_n']=le_degree.fit_transform(inputs['degree'])

    inputs_n=inputs.drop(["company","job","degree"],axis="columns")


    fig,ax=plt.subplots(1,3,sharey=True)
    ax[0].scatter(inputs_n['company_n'],target,marker="+",color="green")
    ax[1].scatter(inputs_n['job_n'],target,marker='+',color="blue")
    ax[2].scatter(inputs_n['degree_n'],target,marker="+",color="red")
    plt.xlabel("company , position ,degree")
    plt.ylabel("target")
    plt.title("Salary more or Less 100k $")
    st.pyplot(fig)

       
