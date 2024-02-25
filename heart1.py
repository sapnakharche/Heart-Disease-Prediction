import streamlit as st
import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.linear_model import LogisticRegression

st.title(""" Heart Disease Pediction System """)
st.write('Please, fill your informations to predict your heart condition')
BMI=st.selectbox("Select your BMI", ("Normal weight BMI  (18.5-25)", "Underweight BMI (< 18.5)" ,"Overweight BMI (25-30)","Obese BMI (> 30)"))
Age=st.selectbox("Select your age",
                            ("18-24", 
                             "25-29" ,
                             "30-34",
                             "35-39",
                             "40-44",
                             "45-49",
                             "50-54",
                             "55-59",
                             "60-64",
                             "65-69",
                             "70-74",
                             "75-79",
                             "55-59",
                             "80 or older"))

Race=st.selectbox("Select your Race", ("Asian",
                             "Black" ,
                             "Hispanic",
                             "American Indian/Alaskan Native",
                             "White",
                             "Other"
                             ))

Gender=st.selectbox("Select your gender", ("Female","Male" ))
Smoking = st.selectbox("Have you smoked more than 100 cigarettes in your entire life ?)",options=("No", "Yes"))
alcoholDink = st.selectbox("How many drinks of alcohol do you have in a week?", options=("No", "Yes"))
stroke = st.selectbox("Did you have a stroke?", options=("No", "Yes"))
sleepTime = st.number_input("Hours of sleep per 24h", 0, 24, 7)
genHealth = st.selectbox("General health",options=("Good","Excellent", "Fair", "Very good", "Poor"))
physHealth = st.number_input("Physical health in the past month (Excelent: 0 - Very bad: 30)", 0, 30, 0)
mentHealth = st.number_input("Mental health in the past month (Excelent: 0 - Very bad: 30)", 0, 30, 0)
physAct = st.selectbox("Physical activity in the past month", options=("No", "Yes"))
diffWalk = st.selectbox("Do you have serious difficulty walking or climbing stairs?", options=("No", "Yes"))
diabetic = st.selectbox("Have you ever had diabetes?",options=("No", "Yes", "Yes, during pregnancy", "No, borderline diabetes"))
asthma = st.selectbox("Do you have asthma?", options=("No", "Yes"))
kidneyDisease= st.selectbox("Do you have kidney disease?", options=("No", "Yes"))
skinCancer = st.selectbox("Do you have skin cancer?", options=("No", "Yes"))

dataToPredic = pd.DataFrame({
   "BMI": [BMI],
   "Smoking": [Smoking],
   "AlcoholDrinking": [alcoholDink],
   "Stroke": [stroke],
   "PhysicalHealth": [physHealth],
   "MentalHealth": [mentHealth],
   "DiffWalking": [diffWalk],
   "Sex": [Gender],
   "AgeCategory": [Age],
   "Race": [Race],
   "Diabetic": [diabetic],
   "PhysicalActivity": [physAct],
   "GenHealth": [genHealth],
   "SleepTime": [sleepTime],
   "Asthma": [asthma],
   "KidneyDisease": [kidneyDisease],
   "SkinCancer": [skinCancer]
 })

# Mapping the data as explained in the script above
dataToPredic.replace("Underweight BMI (< 18.5)",0,inplace=True)
dataToPredic.replace("Normal weight BMI  (18.5-25)",1,inplace=True)
dataToPredic.replace("Overweight BMI (25-30)",2,inplace=True)
dataToPredic.replace("Obese BMI (> 30)",3,inplace=True)

dataToPredic.replace("Yes",1,inplace=True)
dataToPredic.replace("No",0,inplace=True)
dataToPredic.replace("18-24",0,inplace=True)
dataToPredic.replace("25-29",1,inplace=True)
dataToPredic.replace("30-34",2,inplace=True)
dataToPredic.replace("35-39",3,inplace=True)
dataToPredic.replace("40-44",4,inplace=True)
dataToPredic.replace("45-49",5,inplace=True)
dataToPredic.replace("50-54",6,inplace=True)
dataToPredic.replace("55-59",7,inplace=True)
dataToPredic.replace("60-64",8,inplace=True)
dataToPredic.replace("65-69",9,inplace=True)
dataToPredic.replace("70-74",10,inplace=True)
dataToPredic.replace("75-79",11,inplace=True)
dataToPredic.replace("80 or older",13,inplace=True)


dataToPredic.replace("No, borderline diabetes",2,inplace=True)
dataToPredic.replace("Yes (during pregnancy)",3,inplace=True)


dataToPredic.replace("Excellent",0,inplace=True)
dataToPredic.replace("Good",1,inplace=True)
dataToPredic.replace("Fair",2,inplace=True)
dataToPredic.replace("Very good",3,inplace=True)
dataToPredic.replace("Poor",4,inplace=True)


dataToPredic.replace("White",0,inplace=True)
dataToPredic.replace("Other",1,inplace=True)
dataToPredic.replace("Black",2,inplace=True)
dataToPredic.replace("Hispanic",3,inplace=True)
dataToPredic.replace("Asian",4,inplace=True)
dataToPredic.replace("American Indian/Alaskan Native",4,inplace=True)


dataToPredic.replace("Female",0,inplace=True)
dataToPredic.replace("Male",1,inplace=True)

# Load the previously saved machine learning model
filename='finalized_model.sav'
loaded_model= pickle.load(open(filename, 'rb'))
Result=loaded_model.predict(dataToPredic)
ResultProb= loaded_model.predict_proba(dataToPredic)
ResultProb1=round(ResultProb[0][1] * 100, 2)

 # Calculate the probability of getting heart disease
if st.button('PREDICT'):
    #st.write('your prediction:', Result, round(ResultProb[0][1] * 100, 2))
    if (ResultProb1>30):
       st.write('You have a', ResultProb1, '% chance of getting a heart disease' )
    else:
       st.write('You have a', ResultProb1, '% chance of getting a heart disease' )

  
