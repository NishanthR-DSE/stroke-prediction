import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

st.title('Stroke Prediction')
st.markdown('Tell us about yourself')
age=st.slider('Age',1,82,1)
hypertension=st.selectbox('Hypertension',('Yes','No'))
heart_disease=st.selectbox('Heart Disease',('Yes','No'))
ever_married=st.selectbox('Married',('Yes','No'))
work_type=st.selectbox('Work Type',('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
smoking_status=st.selectbox('Smoking Status',('Formerly smoked', 'Never smoked', 'Smokes','Unknown'))
bmi=st.number_input('BMI',10,97)
avg_glucose_level=st.slider('Average Glucose Level',55,271)

with open('Model_Catboost','rb')as file :
    Model=pickle.load(file)
with open ('transform_age','rb') as file:
    age_transformation=pickle.load(file)
with open ('transform_bmi','rb') as file:
    bmi_transformation=pickle.load(file)
with open ('transform_avg_glucose_level','rb') as file:
    avg_glucose_transformation=pickle.load(file)
    

bmi_t=bmi_transformation.transform(pd.DataFrame([bmi]))
avgl=avg_glucose_transformation.transform(pd.DataFrame([avg_glucose_level]))
age_t=age_transformation.transform(pd.DataFrame([age]))

data={'age':age_t[0],
      'hypertension':[0 if hypertension=='No' else 1],
     'heart_disease':[0 if heart_disease=='No' else 1],
     'ever_married':[0 if ever_married=='No' else 1],
     'smoking_status':[0 if smoking_status=='never smoked' else 1 if smoking_status == 'Unknown' else 2 if 
                  smoking_status == 'formerly smoked' else -1],
     'work_type':[0 if work_type == 'Private' else 1 if work_type == 'Selfemployed'
                  else 2 if work_type == 'Govt_job' else 1 if work_type == 'children' else -2],
     'bmi':bmi_t[0],
     'avg_glucose_level':avgl[0]}

predictions =  Model.predict(pd.DataFrame(data))



if st.button('Predict'):
    if predictions == 0:
        st.subheader('You will not get stroke')
    else:
        st.subheader('You will get stroke')
