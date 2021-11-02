# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
# score = svc_model.score(X_train, y_train)

rfc = RandomForestClassifier(n_jobs = -1)
rfc.fit(X_train, y_train)

lr = LogisticRegression()
lr.fit(X_train, y_train)


@st.cache()
def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth, model):
  species = model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  score = model.score(X_train, y_train)
  if species == 0:
    return "Iris-setosa", score
  elif species == 1:
    return "Iris-virginica", score
  else:
    return "Iris-versicolor", score


# st.title("Iris Flower Prediction Web app")
# s_length = st.slider("Sepal Length: ", 5.0, 10.0)
# s_width = st.slider("Sepal Width: ", 3.0, 15.0)
# p_length = st.slider("Petal Length: ", 0.0, 7.0)
# p_width = st.slider("Petal Width: ", 0.5, 7.0)

# if st.button("Predict") :
#     p_out = prediction(s_length, s_width, p_length, p_width)
#     st.write("Species is: ", p_out)
#     st.write("Accuracy is: ", score)

st.sidebar.title("Iris Flower Prediction")

s_length = st.sidebar.slider("Sepal Length", float(iris_df['SepalLengthCm'].min()), float(iris_df['SepalLengthCm'].max()))
s_width = st.sidebar.slider("Sepal Width", float(iris_df['SepalWidthCm'].min()), float(iris_df['SepalWidthCm'].max()))
p_length = st.sidebar.slider("Petal Length", float(iris_df['PetalLengthCm'].min()), float(iris_df['PetalLengthCm'].max()))
p_width = st.sidebar.slider("Petal Width", float(iris_df['PetalWidthCm'].min()), float(iris_df['PetalWidthCm'].max()))

classifier = st.sidebar.selectbox("Choose your Model: ", ("Support Vector Machine", "Random Forest Classifier", "Logistic Regression"))

if st.sidebar.button("Predict") :
  if classifier == "Support Vector Machine" :
    p_out, sc = prediction(s_length, s_width, p_length, p_width, svc_model)
  elif classifier == "Random Forest Classifier" :
    p_out, sc = prediction(s_length, s_width, p_length, p_width, rfc)
  else :
    p_out, sc = prediction(s_length, s_width, p_length, p_width, lr)
  st.write("Species is: ", p_out)
  st.write("Accuracy is: ", sc)

