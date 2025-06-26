# Usage: python TrainGlycoBert.py save_dir max_seq_length num_hidden_layers num_attention_heads hidden_size batch_size num_epochs

import pickle
import torch
from transformers import BertConfig, BertForSequenceClassification
from MSTokenizer import GlycoBertTokenizer
import torch.optim as optim
import time
import numpy as np
import pandas as pd
import sys
import os
from math import ceil

# Get model hyperparameters from command line
save_dir, max_seq_length, num_hidden_layers, num_attention_heads, hidden_size, batch_size, num_epochs = sys.argv[1:8]
max_seq_length = int(max_seq_length)
num_hidden_layers = int(num_hidden_layers)
num_attention_heads = int(num_attention_heads)
hidden_size = int(hidden_size)
batch_size = int(batch_size)
num_epochs = int(num_epochs)

print(f"max_seq_length = {max_seq_length}")
print(f"num_hidden_layers = {num_hidden_layers}")
print(f"num_attention_heads = {num_attention_heads}")
print(f"hidden_size = {hidden_size}")
print(f"batch_size = {batch_size}")
print(f"num_epochs = {num_epochs}")

# Set directories 
workingDir = './'
saveDir = ''.join([workingDir, save_dir])
print(f"Output will be saved in = {saveDir}")

# Initiate tokenizer
vocab_path = ''.join([workingDir, 'tokenizer/vocab.json'])
tokenizer = GlycoBertTokenizer.load_vocabulary(path=vocab_path)

# Load train tensor and labels
train_file = ''.join([workingDir, 'full_tensor.pt'])
train_tensor = torch.load(train_file)

labels_file = ''.join([workingDir, 'label.pkl'])
with open(labels_file, 'rb') as file:
    labels = pickle.load(file)

input_ids = train_tensor["token_ids"]
attention_masks = train_tensor["attention_mask"]
labels = torch.tensor(labels)
    
# Create GlycoBert model
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    hidden_size=hidden_size,
    intermediate_size=4*hidden_size,
    max_position_embeddings=max_seq_length,
    num_labels=3590
)

model = BertForSequenceClassification(config)

# Check if multiple GPUs are available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.device_count()} GPUs for training")
else:
    device = torch.device("cpu")
    print("Using CPU for training")

# Use DataParallel to use multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

# Rest of your training script
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_function = torch.nn.CrossEntropyLoss()

data = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    total_samples = 0
    correct_predictions = 0
    start_time = time.time()  # Start the timer for the epoch
    model.train()
    
    for batch in dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = batch_attention_masks.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            labels=batch_labels
        )

        logits = outputs.logits
        loss = loss_function(logits, batch_labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_labels.size(0)
        total_samples += batch_labels.size(0)
        correct_predictions += torch.sum(torch.argmax(logits, dim=1) == batch_labels).item()

    # Calculate average metrics for the epoch
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    
    end_time = time.time()  # Stop the timer for the epoch
    epoch_time = end_time - start_time  # Calculate the time for the epoch

    # Print or log the metrics and time for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f} - Accuracy: {accuracy:.4f} - Time: {epoch_time:.2f} seconds")
    
    # Save the trained model
    output_dir = os.path.join(saveDir, f'output_e{epoch+1}')

    # Check if model is wrapped with DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    # Save the model
    model_to_save.save_pretrained(output_dir)    