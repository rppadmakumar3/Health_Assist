import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add a header/banner
st.title("Health ChatBot")

# Create a text input for user messages with some styling
user_input = st.text_area("You:", "", key="user_input", height=100)

if user_input:
    # Encode the user's message
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response using the GPT-2 model
    with st.spinner("ChatBot is thinking..."):
        response_ids = model.generate(input_ids, num_return_sequences=1)

    # Decode and display the chatbot's response
    chatbot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    # Display user message and chatbot response in chat history
    st.text("User: " + user_input)
    st.text("HealthAssist: " + chatbot_response)
