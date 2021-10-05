import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("kaggle_diabetes.csv")
#list_feature_contain_zero
list_feature_contain_zero=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#Split dataset into train and test here only to avoid data leakage
x= df.drop(columns='Outcome')
y=df['Outcome']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

#Function to calculate meand and median TRain data
def impute_zero_tain_data(x_train,list_feature_contain_zero):
    for feature in list_feature_contain_zero:
        
        if list_feature_contain_zero=='SkinThickness' or  list_feature_contain_zero == 'Insulin' :
            x_train[list_feature_contain_zero]=x_train[list_feature_contain_zero].fillna(x_train[list_feature_contain_zero].median())
        else:
            x_train[list_feature_contain_zero]=x_train[list_feature_contain_zero].fillna(x_train[list_feature_contain_zero].mean())
            
            
            
impute_zero_tain_data(x_train,list_feature_contain_zero)


#Function to calculate meand and median Test Data
def impute_zero_test_data(x_test,list_feature_contain_zero):
    for feature in list_feature_contain_zero:
        
        if list_feature_contain_zero=='SkinThickness' or  list_feature_contain_zero == 'Insulin' :
            x_test[list_feature_contain_zero]=x_test[list_feature_contain_zero].fillna(x_train[list_feature_contain_zero].median())
        else:
            x_test[list_feature_contain_zero]=x_test[list_feature_contain_zero].fillna(x_train[list_feature_contain_zero].mean())
            
            
            
impute_zero_test_data(x_test,list_feature_contain_zero)

#Model Building
#using Random Forest model comes under Ensamble technique under Bootstrap Aggregetion


classifierr=RandomForestClassifier(n_estimators=20)
classifierr.fit(x_train,y_train)


#pred=classifier.predict(x_test)


#from sklearn.metrics import classification_report,confusion_matrix
#print(confusion_matrix(y_test,pred))
#print(classification_report(y_test,pred))

filename='diabetic_predict_rfc_model.pkl'
pickle.dump(classifierr,open(filename,'wb'))
