import streamlit as st
from PIL import Image
import requests
import matplotlib.pyplot as plt
import os
import io

st.title('Pill Visual Quality Checker')

ipex_tab = st.tabs(["Intel Extension for PyTorch"])

# Start of Intel Extension for PyTorch Implementation

st.divider()

st.markdown('#### Model Training')

data_folder = st.text_input('Root Training Data Folder', value='./store/datasets/medication_qaqc/data/')
model_path = st.text_input('Save Model Path', value='./store/models/medication_qaqc/model_ipex.h5')
neg_class = st.number_input('Passing Quality Label', min_value=0, max_value=100)
learning_rate = st.slider('Learning Rate', min_value=0.0001, max_value=0.05, step=.0001, value=.025, format='%f')
epochs = st.number_input('Epochs', min_value=1, max_value=100, step=1, value=5)
data_aug = st.checkbox('Augment Training Data')

if st.button('Train Model', key='ipex training'):
    # build request

    URL = 'http://medication_qaqc:5002/train-ipex'
    DATA = {"data_folder": data_folder, "neg_class": neg_class, "modeldir": model_path,
            "learning_rate": learning_rate, "epochs": epochs, "data_aug": data_aug}
    TRAINING_RESPONSE = requests.post(url=URL, json=DATA)

    if len(TRAINING_RESPONSE.text) < 40:
        st.error("Model Training Failed")
        st.info(TRAINING_RESPONSE.text)
    else:
        st.success('Training was Successful')
        st.info('Trained Model Location: ' + str(TRAINING_RESPONSE.json().get('Model Location')))

st.divider()

st.markdown('#### Intel Extension for PyTorch Inference')

data_folder = st.text_input('Root Store', value='./store/datasets/medication_qaqc/data/')
trained_model_path = st.text_input('Trained Model Path', value='./store/models/medication_qaqc/model_ipex.h5')
batch_size = st.number_input('Inference Batch Size', min_value=5, max_value=100)

if st.button('Start Batch Evaluation', key='ipex inference'):
    # build request
    URL = 'http://medication_qaqc:5002/predict-ipex'
    DATA = {"trained_model_path": trained_model_path, "data_folder": data_folder, "batch_size": batch_size}
    INFERENCE_RESPONSE = requests.post(url=URL, json=DATA)

    if len(INFERENCE_RESPONSE.text) < 40:
        st.error("Inference Failed")
        st.info(INFERENCE_RESPONSE.text)
    else:
        st.success('Inference was Successful')
        classified_pills = INFERENCE_RESPONSE.json().get('Prediction Output')

        with st.expander("Pill Classifications"):
            for label, file in classified_pills:
                if label == 1.0:
                    st.markdown('### Faulty Pill')
                    st.text(os.path.join(data_folder, 'blind/', file))
                    image = Image.open(os.path.join(data_folder, 'blind/', file))
                    st.image(image, width=800)
                else:
                    st.markdown('### Valid Pill')
                    st.text(os.path.join(data_folder, 'blind/', file))
                    image = Image.open(os.path.join(data_folder, 'blind/', file))
                    st.image(image, width=800)
