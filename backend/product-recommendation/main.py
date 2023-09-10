# Import fastapi, numpy, pandas, sklearn and os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import os

# Create a fastapi app
app = FastAPI()

# Define a class for the request body
class UserInput(BaseModel):
    product_name: str

# Define a class for the response
class Prediction(BaseModel):
    products: list[str]

# Read the csv file
products = pd.read_csv('flipkart_com-ecommerce_sample.csv')

# Create a tf-idf matrix
products['description'] = products['description'].fillna('')
tfv = TfidfVectorizer(max_features=None,
                     strip_accents='unicode',
                     analyzer='word',
                     min_df=10,
                     token_pattern=r'\w{1,}',
                     ngram_range=(1,3),
                     stop_words='english')
tfv_matrix = tfv.fit_transform(products['description'])

# Compute a sigmoid kernel
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)

# Create an index for the products
indices = pd.Series(products.index,index=products['product_name']).drop_duplicates()

# Define a function for product recommendation
def product_recommendation(title, sig=sig):
    """
    Receives a product name as a string and returns a list of recommended products based on the sigmoid kernel similarity.
    """
    # Get the index of the product
    indx = indices[title]
    # Get the similarity scores of the product with other products
    sig_scores = list(enumerate(sig[indx]))
    # Sort the similarity scores in descending order
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    # Get the top 10 most similar products
    sig_scores = sig_scores[1:11]
    # Get the indices of the top 10 most similar products
    product_indices = [i[0] for i in sig_scores]
    # Return the product names of the top 10 most similar products
    return products['product_name'].iloc[product_indices]

# Create a route for prediction
@app.post('/predict', response_model=Prediction)
def predict(user_input: UserInput):
    """
    Receives a user input as a string and returns a prediction as a list of recommended products using a pickle model.
    """
    # Get the user input from the request body as string
    user_input = user_input.product_name
    # Try to make predictions using the user input
    try:
        prediction = product_recommendation(user_input)
        # Convert the prediction to a list of strings
        prediction = prediction.tolist()
        # Return the prediction as json
        return {'products': prediction}
    except KeyError as e:
        # Raise an exception if the user input is not in the index
        raise HTTPException(status_code=404, detail=f'Product not found: {e}')
