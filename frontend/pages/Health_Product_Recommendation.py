# Import streamlit and requests
import streamlit as st
import requests

# Create a title and a sidebar
st.title('Health Product Recommendation System')

# Create a text input for the user
st.subheader('Product Recommendation')
user_input = st.text_input('Enter the name of the product')

# Create a button to make predictions
if st.button('Predict'):
    # Send a post request to the backend with the user input as json data
    response = requests.post('http://product-recommendation:8006/predict', json={'product_name': user_input})
    # Check if the response status code is 200 (OK)
    if response.status_code == 200:
        # Get the prediction from the response as json
        prediction = response.json()
        # Display the prediction as a list of recommended products
        st.write(f'Top Recommended products are:')
        # Format the prediction as a bullet list
        for product in prediction['products']:
            st.write(f'- {product}')
    else:
        # Display an error message if the response status code is not 200 (OK)
        st.error(f'An error occurred: {response.text}')
