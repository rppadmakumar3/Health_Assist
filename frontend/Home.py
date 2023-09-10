# main.py
import streamlit as st
from pages import Health_Product_Recommendation, Chatbot, Disease_Prediction, Medical_Imaging_Processing, Pill_Quality

def main():
    st.title("HealthAssist - Your Personal Health Guardian")
    st.markdown("---------------------------------------")

    st.image("/home/ubuntu/HealthAssist/Assets/img_1.jpg")

if __name__ == "__main__":
    main()
    