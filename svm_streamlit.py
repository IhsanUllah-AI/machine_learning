import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot  as plt
from sklearn.datasets import load_iris
import joblib

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
df['flower name']=df.target.apply(lambda x:iris.target_names[x])
from streamlit_option_menu import option_menu
with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        menu_icon="cast",
        options=['Home','prediction','contact','About'],
        icons=['house','bar-chart','envelope','people'],
        default_index=0,
    )

if selected=="Home":
    st.title("Flower Type Prediction System")
    st.info("Sample")
    st.write(df)
    
    st.warning("Data visulaiztion")
    df0=df[df['target']==0]
    df1=df[df['target']==1]
    df2=df[df['target']==2]
    fig,ax=plt.subplots()
    ax.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],marker='+',color='green',label='setosa')
    ax.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],marker='+',color='blue',label='varsicolor')
    ax.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],marker='+',color='yellow',label='verginca ')
    plt.xlabel("sepal length (cm)")
    plt.ylabel("sepal width (cm)")
    plt.legend()
    st.pyplot(fig)
    
    fig,ax=plt.subplots()
    ax.scatter(df0['petal length (cm)'],df0['petal width (cm)'],marker='+',color='green',label='setosa')
    ax.scatter(df1['petal length (cm)'],df1['petal width (cm)'],marker='+',color='blue',label='varsicolor')
    ax.scatter(df2['petal length (cm)'],df2['petal width (cm)'],marker='+',color='red',label='verginca ')
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    plt.legend()
    st.pyplot(fig)

elif selected=="prediction":
    model=joblib.load("svm_model")

    st.info("Sepal Data entry ")
    col1,col2=st.columns(2)
    with col1:
        sep_len=st.number_input("Enter sepal length in cm (not exceed from 8 cm)",step=0.1) 
    with col2:
         sep_width=st.number_input("Enter sepal width in cm (not exceed from 4.5)",step=0.1)

    st.info("Petal Data Entry")
    col3,col4=st.columns(2)
    with col3:
        pet_len=st.number_input("Enter pepal length in cm (not exceed from 7.0))",step=0.1) 
    with col4:
        pe_width=st.number_input("Enter pepal width in cm ((not exceed from 2.6))",step=0.1) 

    if st.button("predict"):
        y_pred=model.predict(np.array([[sep_len,sep_width,pet_len,pe_width]]))
        if y_pred==0:
            st.write("""# Prediction Result """)
            st.write("### Setosa")
            st.write("""# Description
Iris Setosa is the smallest of the three species and is easily recognizable
by its petite petals. The species has blue or violet flowers, and it is commonly
found in northern regions and cooler climates. The flowers are usually around 
5 cm across and have a striking appearance due to their vibrant color.""")
            st.write("""### Key Measurment
Sepal Length: 4.3 – 5.8 cm\n
Sepal Width: 2.3 – 4.4 cm\n
Petal Length: 1.0 – 1.9 cm\n
Petal Width: 0.1 – 0.6 cm""")
            
        elif y_pred==1:
            st.write("""# Prediction Result """)
            st.write("### Versicolor")
            st.write("""# Description
Iris Versicolor, also known as the "Blue Flag" or "Harlequin Blueflag," is a medium-sized species with a striking 
violet-blue to purple flower. The species is native to North America and is often found in wetlands or along stream
banks. The petals are more substantial compared to Setosa but not as large as those of Virginica.""")
            st.write("""### Key Measurment
Sepal Length: 4.9 – 7.0 cm\n
Sepal Width: 2.0 – 3.4 cm\n
Petal Length: 3.0 – 5.1 cm\n
Petal Width: 1.0 – 1.8 cm""")

        elif y_pred==2:
            st.write("""# Prediction Result """)
            st.write("### Verginica")
            st.write("""# Description
Iris Virginica, also known as the "Virginia Iris" or "Southern Blue Flag," is the largest of the three species.
It features deep blue to purple flowers with large, broad petals. This species is typically found in the eastern
United States, thriving in wetlands and marshy areas.""")
            st.write("""### Key Measurment
Sepal Length: 4.9 – 7.9 cm\n
Sepal Width: 2.2 – 3.8 cm\n
Petal Length: 4.5 – 6.9 cm\n
Petal Width: 1.4.0 – 2.5 cm""")


elif selected=='contact':
    st.write("# visit to my portfolio and contact me through email or phone or whatsapp or through linkdn")
    st.markdown("[Portfolio](https://ihsanullah.netlify.app)")


elif selected=="About":
    st.info("""### Flower Type Prediction Project 
This project involves building a machine learning model 
to  predict the species of iris flowers—Setosa, Versicolor, or Virginica—based on measurements 
of sepal length, sepal width, petal length, and petal width, all in centimeters.Using a dataset
of labeled iris flowers, the model is trained to identify patterns and relationships between the 
features and the corresponding flower species. The project demonstrates the application of 
classification algorithms in machine learning, showcasing how input data can lead to accurate 
predictions in real-world scenarios. The model's effectiveness can be evaluated through metrics
such as accuracy and confusion matrix,providing insights into its performance and reliability. """)