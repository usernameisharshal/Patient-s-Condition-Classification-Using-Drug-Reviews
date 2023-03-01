# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv("C:\\Users\\abhis\\OneDrive\\Desktop\\filtered_df.csv")

# Clean the data
df = df.dropna()
df = df.drop_duplicates()
df["rating"] = df["rating"].astype(int)

# Create the feature matrix
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
reviews = vectorizer.fit_transform(df["review"])

# Train the model
model = LogisticRegression(multi_class="ovr")
model.fit(reviews, df["condition"])

# Write a function that returns the predicted condition and recommended drugs
def predict_condition(review):
    review = vectorizer.transform([review])
    condition = model.predict(review)[0]
    return condition


# Write a Streamlit app that allows the user to enter a review and receive recommendations
st.title("Patient's Condition Classification Using Drug Reviews")
review = st.text_input("Enter a patient review:")

if st.button("Predict Condition"):
    condition = predict_condition(review)
    st.write("Predicted Condition:", condition)



