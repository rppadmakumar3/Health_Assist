import streamlit as st
from transformers import Conversation, pipeline

# Create a chatbot UI
st.title("DocBot")

# Initialize the pipeline
pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Create an empty conversation to start
conversation = Conversation()

# Get user input from the user
user_input = st.text_input("User", "Type your message here")

# Add the user's message to the conversation
conversation.add_user_input(user_input)

# Check if the user input is not empty
if user_input:
    # Get the bot's response
    bot_output = pipe(conversation)
    
    # Get the generated text from the bot's response
    bot_response = bot_output.generated_responses[0]
    
    # Display the bot's response
    st.text("Health Bot: " + bot_response)