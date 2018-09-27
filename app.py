# this is the python script for a program for hand written digit recognition systerm, using scikit learn
import time as time
import numpy as np #used for mathematical computing
import pandas as pd # used for using the dataset
import matplotlib.pyplot as plt # used for the data visualizations
import seaborn as sns # acts as  a wrapper for matplotlib, it basically makes a longer code one or two line syntax
# loading the dataset
df = pd.read_csv("/home/achintya/digitRecognizer/train.csv") #used to read the scv dataset
print("printing the the 5 rows of data")
time.sleep(3)
print(df.head()) # prints the first 5 rows of the dataset
print("printing the values of df")
time.sleep(3)
print(df.values)
time.sleep(3)
print("printing the example of the dataset of handwritten objects")
r = 5 
c = 5 
t =r*c
fig = plt.figure(figsize=(r,c)) # accmodating multiple images in one figure 
for i in range(0,t):
	pic = df.values[i][1:].reshape(28,28)
	fig.add_subplot(r,c,i+1)
	plt.imshow(pic,cmap='binary')
plt.show() #used to display the figure
df = df[:10000] #using only the first 10000 examples from the dataset
print("the dimesions of the dataset is", df.shape)
X = df.drop(['label'],axis =1) 
print(X.shape)
y = df['label']
print("the set of labels for supervised learning are")
time.sleep(3)
print(y)
from sklearn.model_selection import train_test_split 
# the above line divides the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
# test is 33% and training is 67%
# random state is the randomizing order 
from sklearn.metrics import classification_report
print("we are using the Logistic Regression algorithm for training the model")
# the model takes good amount of time to train, so be ready for waiting for 5 mins atleast 
# when you run this code
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter= 100)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(classification_report(y_test, prediction))
