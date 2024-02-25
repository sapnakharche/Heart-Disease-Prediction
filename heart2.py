import numpy as np
import pandas  as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("heart_2020_cleaned.csv")
print(df.info())
duplicateObser = df[df.duplicated()]
LabelsDupObser=duplicateObser.axes[0].tolist()
print('Number of duplicated observations:', duplicateObser.shape[0])
df=df.drop_duplicates()
print(df.shape)

df.replace("Yes",1,inplace=True)
df.replace("No",0,inplace=True)

target=df["HeartDisease"]
df.drop(["HeartDisease"], axis=1, inplace=True)
df.AgeCategory.unique()
df.replace("18-24",0,inplace=True)
df.replace("25-29",1,inplace=True)
df.replace("30-34",2,inplace=True)
df.replace("35-39",3,inplace=True)
df.replace("40-44",4,inplace=True)
df.replace("45-49",5,inplace=True)
df.replace("50-54",6,inplace=True)
df.replace("55-59",7,inplace=True)
df.replace("60-64",8,inplace=True)
df.replace("65-69",9,inplace=True)
df.replace("70-74",10,inplace=True)
df.replace("75-79",11,inplace=True)
df.replace("80 or older",13,inplace=True)

df.Diabetic.unique()
df.replace("No, borderline diabetes",2,inplace=True)
df.replace("Yes (during pregnancy)",3,inplace=True)

df.GenHealth.unique()
df.replace("Excellent",0,inplace=True)
df.replace("Good",1,inplace=True)
df.replace("Fair",2,inplace=True)
df.replace("Very good",3,inplace=True)
df.replace("Poor",4,inplace=True)

df.Race.unique()
df.replace("White",0,inplace=True)
df.replace("Other",1,inplace=True)
df.replace("Black",2,inplace=True)
df.replace("Hispanic",3,inplace=True)
df.replace("Asian",4,inplace=True)
df.replace("American Indian/Alaskan Native",4,inplace=True)

df.Gender.unique()
df.replace("Female",0,inplace=True)
df.replace("Male",1,inplace=True)

df['BMI'].mask(df['BMI']  < 18.5, 0, inplace=True)
df['BMI'].mask(df['BMI'].between(18.5,25), 1, inplace=True)
df['BMI'].mask(df['BMI'].between(25,30), 2, inplace=True)
df['BMI'].mask(df['BMI']  > 30, 3, inplace=True)

X_train,X_test,y_train,y_test = train_test_split(df,target,test_size=50,random_state=2)

LogRegModel=LogisticRegression()
LogRegModel.fit(X_train, y_train)
#with open('LogRegModel.pkl', 'wb') as f:
 #   pickle.dump(LogRegModel, f)
filename = "finalized_model.sav"
pickle.dump(LogRegModel, open(filename, "wb"))