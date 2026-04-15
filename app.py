from google.colab import files 
uploaded = files.upload() 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder 
df = pd.read_csv("dataset.csv") 
print(df.head()) 
print(df.columns) 
if len(df.columns) == 1: 
    df = df[df.columns[0]].str.split(",", expand=True) 
df.columns = ['Age','12th_Percentage','BTech_Stream','CGPA','Interested_Field','Career'] 
print(df.head()) 
df[['Age','12th_Percentage','CGPA']] = df[['Age','12th_Percentage','CGPA']].astype(float) 
from sklearn.preprocessing import LabelEncoder 
le_stream = LabelEncoder() 
le_interest = LabelEncoder() 
le_career = LabelEncoder() 
df['BTech_Stream'] = le_stream.fit_transform(df['BTech_Stream']) 
df['Interested_Field'] = le_interest.fit_transform(df['Interested_Field']) 
df['Career'] = le_career.fit_transform(df['Career']) 
from sklearn.tree import DecisionTreeClassifier 
X = df[['Age','12th_Percentage','BTech_Stream','CGPA','Interested_Field']] 
y = df['Career'] 
model = DecisionTreeClassifier()
model.fit(X, y) 
age = int(input("Enter Age: ")) 
marks = float(input("Enter 12th Percentage: ")) 
stream = input("Enter BTech Stream (CSE/IT/ECE): ") 
 
cgpa = float(input("Enter CGPA: ")) 
interest = input("Enter Interested Field: ") 
stream_encoded = le_stream.transform([stream])[0] 
interest_encoded = le_interest.transform([interest])[0] 
prediction = model.predict([[age, marks, stream_encoded, cgpa, interest_encoded]]) 
result = le_career.inverse_transform(prediction) 
print("       Recommended Career:", result[0])
