from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load the fine-tuned tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_shakespeare_gpt2')
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_shakespeare_gpt2')

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate a response based on user input
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)  # Move input to GPU
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Store chat history
chat_history = []

# Flask route for the homepage
@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history
    
    if request.method == "POST":
        user_input = request.form.get("user_input")
        
        # If user clicks "Retry" without input, just regenerate the last response
        if user_input == "":
            user_input = chat_history[-2]['user']  # Take the last user input

        # Generate the model's response
        model_response = generate_text(user_input)

        # Update chat history
        chat_history.append({"user": user_input, "bot": model_response})

    return render_template("chat.html", chat_history=chat_history)

# Route to reset chat history
@app.route("/reset")
def reset():
    global chat_history
    chat_history = []
    return redirect(url_for('chat'))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
