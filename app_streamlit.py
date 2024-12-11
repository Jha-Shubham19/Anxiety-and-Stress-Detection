import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load the trained model from the checkpoint directory
model = DistilBertForSequenceClassification.from_pretrained('checkpoint-717')  # Change to your checkpoint number
# Load the tokenizer again if needed
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Define the mental health statements
mental_health_statements = ["Stress", "Depression", "Anxiety", "Normal"]

# Streamlit UI
st.title("Mental Health Condition Predictor")
st.write("Enter a statement about your feelings and get a prediction of your mental health condition.")

# Text input with increased height
input_text = st.text_area("Your Statement:", "swinging from being really happy to just feeling empty for days", height=200)


# Button to predict
if st.button("Predict"):
    if input_text:
        # Tokenize the input
        inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=128, return_tensors='pt')

        # Get the model predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        predictions = torch.argmax(outputs.logits, dim=-1)
        predictions_value = predictions.item()
        
        # Display the result
        st.success(f"Predicted class: {predictions_value}, Condition: {mental_health_statements[predictions_value]}")
    else:
        st.error("Please enter a statement.")



# I feel like something bad is about to happen
# I'm completely overwhelmed with deadlines.
# Sun is only star in solar system
# I thought that we were meant to be

# You took my heart and made it bleed
# I gave you all my ecstasy
# I know you'll be the death of me
# Left lipstick on my Hennessy; felt like you took my soul from me
# You gave me all your ecstasy; I thought that we were meant to be
# Your love is suicidal
# For me, your love is suicidal

# When computers came to India, everybody used to say like this:
# "All jobs will be lost, computers will do everything, what will people do?".
# Rajiv Gandhi was abused a lot. But now in the end everybody is just using computers.
# It has just made jobs easier. And simple repetitive tasks have been eliminated, which do not require any brain power.
# People have become more creative and open. The same thing will happen with AI. Those who are doing menial jobs will be eliminated. Those who are creative, who are using their brains will remain

# I feel like my mind is constantly racing, filled with thoughts that I just can’t control. 
# It’s like I’m waiting for something bad to happen, even when there’s no real reason for it. 
# No matter how hard I try to focus or calm down, 
# there’s this nagging sense of worry that just won’t go away.

# Today was just an ordinary day, nothing too exciting but nothing too dull either. 
# I went through my usual routine, grabbing my morning coffee and settling into work. 
# There were a few moments of laughter with colleagues, and some quiet time during lunch where I enjoyed the break.
# The weather was nice, not too hot or cold, which made the walk back home pleasant. 
# Now, as the evening winds down, I feel content and ready to relax before doing it all again tomorrow.

# I've been feeling so stressed lately because I have this NLP assignment that’s still pending, 
# and the deadline is fast approaching. 
# No matter how much I try to focus, 
# I just can’t seem to get through all the material and complete the tasks. 
# Every time I think about it, my mind starts racing with all the things I still need to do. 
# It’s like there’s this constant pressure weighing on me, making it hard to concentrate on anything else. 
# I just hope I can finish it on time without completely burning out.


