import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load your text files into a dataset
dataset = load_dataset('text', data_files={'train': ['Hamletplay.txt', 'Macbethplay.txt']})

# 2. Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 2.1 Add a pad token to the GPT-2 tokenizer
tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the pad token

# 3. Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 4. Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    # Shift the input tokens to create labels
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 5. Set up fine-tuning configurations
training_args = TrainingArguments(
    output_dir="./fine_tuned_shakespeare",  # Directory to save the fine-tuned model
    overwrite_output_dir=True,
    num_train_epochs=3,                 # Number of epochs for training
    per_device_train_batch_size=4,      # Adjust this based on your GPU memory
    save_steps=1000,                    # Save every 1000 steps
    save_total_limit=2,                 # Only keep the last 2 checkpoints
)

# 6. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

# 7. Train the model
trainer.train()

# 8. Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_shakespeare_gpt2")
tokenizer.save_pretrained("./fine_tuned_shakespeare_gpt2")
