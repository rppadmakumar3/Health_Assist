import streamlit as st
import requests

# Define a mapping between labels and disease names
label_to_disease = {
    "LABEL_0": "(Vertigo) Paroxysmal Positional Vertigo",
    "LABEL_1": "AIDS",
    "LABEL_2": "Acne",
    "LABEL_3": "Alcoholic hepatitis",
    "LABEL_4": "Allergy",
    "LABEL_5": "Arthritis",
    "LABEL_6": "Bronchial Asthma",
    "LABEL_7": "Cervical spondylosis",
    "LABEL_8": "Chicken pox",
    "LABEL_9": "Chronic cholestasis",
    "LABEL_10": "Common Cold",
    "LABEL_11": "Dengue",
    "LABEL_12": "Diabetes",
    "LABEL_13": "Dimorphic hemorrhoids (piles)",
    "LABEL_14": "Drug Reaction",
    "LABEL_15": "Fungal infection",
    "LABEL_16": "GERD",
    "LABEL_17": "Gastroenteritis",
    "LABEL_18": "Heart attack",
    "LABEL_19": "Hepatitis B",
    "LABEL_20": "Hepatitis C",
    "LABEL_21": "Hepatitis D",
    "LABEL_22": "Hepatitis E",
    "LABEL_23": "Hypertension",
    "LABEL_24": "Hyperthyroidism",
    "LABEL_25": "Hypoglycemia",
    "LABEL_26": "Hypothyroidism",
    "LABEL_27": "Impetigo",
    "LABEL_28": "Jaundice",
    "LABEL_29": "Malaria",
    "LABEL_30": "Migraine",
    "LABEL_31": "Osteoarthritis",
    "LABEL_32": "Paralysis (brain hemorrhage)",
    "LABEL_33": "Peptic ulcer disease",
    "LABEL_34": "Pneumonia",
    "LABEL_35": "Psoriasis",
    "LABEL_36": "Tuberculosis",
    "LABEL_37": "Typhoid",
    "LABEL_38": "Urinary tract infection",
    "LABEL_39": "Varicose veins",
    "LABEL_40": "Hepatitis A"
}

def show_page():
    st.title("Disease Prediction")

    # User input for epochs
    epochs = st.number_input("Number of Epochs", min_value=1, step=1)

    # Train Model button
    if st.button("Train Model"):
        if epochs >= 1:
            st.info(f"Training the model for {epochs} epochs...")

            # Send a request to the backend to initiate training
            response = send_train_request(epochs)

            # Display the backend's response (you can customize this)
            st.success(response)
        else:
            st.error("Please choose a valid number of epochs.")

    # User input for symptoms
    user_input = st.text_area("Enter symptoms:", "")

    # Predict Disease button
    if st.button("Predict Disease"):
        if user_input:
            st.info("Predicting disease...")

            # Send a request to the backend for prediction
            prediction_response = send_prediction_request(user_input)

            # Display the prediction in a table format
            if prediction_response:
                st.subheader("Prediction Result")
                predicted_label = prediction_response['label']
                disease_name = label_to_disease.get(predicted_label, "Unknown Disease")
                confidence = prediction_response['confidence']

                # Create a table to display the prediction result
                result_table = {
                    "Predicted Disease": [disease_name],
                    "Confidence": [confidence]
                }
                st.table(result_table)
            else:
                st.error("Prediction failed. Please try again later.")

def send_train_request(epochs):
    # Define the backend API URL
    backend_url = "http://disease_prediction:8000/train"  # Replace with your backend URL

    try:
        # Send a POST request to the backend with 'epochs' as a query parameter
        response = requests.post(backend_url, params={"epochs": epochs})

        # Check the response status code
        if response.status_code == 200:
            return "Model training started successfully."
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

def send_prediction_request(input_text):
    # Define the backend API URL for prediction
    backend_predict_url = "http://disease_prediction:8000/predict_disease"  # Replace with your backend URL

    try:
        # Send a POST request to the backend with 'text' as a JSON payload
        response = requests.post(backend_predict_url, json={"text": input_text})

        # Check the response status code
        if response.status_code == 200:
            return response.json()
        else:
            return None

    except requests.exceptions.RequestException as e:
        return None

if __name__ == "__main__":
    show_page()
