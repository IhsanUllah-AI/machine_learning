import streamlit as st
import joblib
import numpy as np

#load model
model=joblib.load("joblib_model")
st.sidebar.title("saalry prediction")

st.title("salary prection base on experince")

inp=st.number_input("enter year of experinec to preidct salary ",step=0.1,)

if st.button("predict salary"):
    if inp < 0:
       st.info("experince can not be nagative")
       st.warning("please enter value  above zero ")
    else:
        prediction=model.predict(np.array([[inp]]))
        st.success(f"predicted salary has {prediction}")
        st.balloons()
      

