# Usage: python3 TrainGlycoBart.py save_dir num_encoder_layers num_decoder_layers num_attention_heads dim_model batch_size num_epochs 

import sys
import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from MSTokenizer import GlycanTranslationData
from MSTokenizer import GlycoBartTokenizer
from transformers import BartForConditionalGeneration, BartConfig

# Get model hyperparameters from command line
save_dir, num_encoder_layers, num_decoder_layers, num_attention_heads, d_model, batch_size, num_epochs = sys.argv[1:8]
num_encoder_layers = int(num_encoder_layers)
num_decoder_layers = int(num_decoder_layers)
num_attention_heads = int(num_attention_heads)
d_model = int(d_model)
batch_size = int(batch_size)
num_epochs = int(num_epochs)

print(f"num_encoder_layers = {num_encoder_layers}")
print(f"num_decoder_layers = {num_decoder_layers}")
print(f"num_attention_heads = {num_attention_heads}")
print(f"model_dimension = {d_model}")
print(f"batch_size = {batch_size}")
print(f"num_epochs = {num_epochs}")

# Set directories
workingDir = "./" #working directory
saveDir = ''.join([workingDir, save_dir])
print(f"Output will be saved in = {saveDir}")

# Initiate tokenizer
vocab_file = ''.join([workingDir,'tokenizer/vocab.json'])
tokenizer = GlycoBartTokenizer.load_vocabulary(path=vocab_file)
vocab_size = tokenizer.vocab_size

# Load input and output tokens
input_corpus_path = ''.join([workingDir,'input_corpus.pt'])
output_corpus_path = ''.join([workingDir,'output_corpus.pt'])
input_corpus = torch.load(input_corpus_path)
output_corpus = torch.load(output_corpus_path)

pad_token_id = tokenizer.vocab[tokenizer.special_tokens['pad_token']]
eos_token_id = tokenizer.vocab[tokenizer.special_tokens['eos_token']]

dataset = GlycanTranslationData(input_corpus, output_corpus, pad_token_id, eos_token_id)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create GlycoBart model
config = BartConfig(
    # Basic configuration
    vocab_size=vocab_size,          # Your vocabulary size
    d_model=d_model,               # Size of embeddings
    num_attention_heads=num_attention_heads,    # Number of attention heads
    num_encoder_layers=num_encoder_layers,     # Number of encoder layers
    num_decoder_layers=num_decoder_layers,     # Number of decoder layers
    encoder_ffn_dim=4*d_model,      # Dimension of encoder feed-forward networks
    decoder_ffn_dim=4*d_model,      # Dimension of decoder feed-forward networks

    # Positional encodings
    max_position_embeddings=tokenizer.max_seq_length,   

    # Regularization
    dropout_rate=0.1,          # Dropout rate for BART

    # Special tokens - if you have defined special tokens in your tokenizer, you need to configure their ids here
    bos_token_id=tokenizer.vocab[tokenizer.special_tokens['bos_token']],
    eos_token_id=tokenizer.vocab[tokenizer.special_tokens['eos_token']],
    pad_token_id=tokenizer.vocab[tokenizer.special_tokens['pad_token']],
    unk_token_id=tokenizer.vocab[tokenizer.special_tokens['unk_token']],
    mask_token_id=tokenizer.vocab[tokenizer.special_tokens['mask_token']],
    decoder_start_token_id=tokenizer.vocab[tokenizer.special_tokens['eos_token']]
    )

model = BartForConditionalGeneration(config)

# Use DataParallel to use multiple GPUs

# Check if multiple GPUs are available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.device_count()} GPUs for training")
else:
    device = torch.device("cpu")
    print("Using CPU for training")

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

# Define the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training
num_epochs = num_epochs
model.train()

# Training loop
for epoch in range(num_epochs):
    start_time_epoch = time.time()  # Start time for the epoch
    total_loss = 0.0

    for batch in dataloader:

        optimizer.zero_grad()

        # Ensure each tensor in the batch is on the correct device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass and compute loss
        outputs = model(**batch)
        loss = outputs.loss
        
        # Sum the loss if using DataParallel
        if torch.cuda.device_count() > 1:
            loss = loss.sum()

        total_loss += loss.item()  

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

    # Calculate average loss and elapsed time for the epoch
    avg_loss = total_loss / len(dataloader)
    elapsed_time_epoch = time.time() - start_time_epoch

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - Time: {elapsed_time_epoch:.2f}s")

    # Save the trained model
    save_directory = os.path.join(saveDir, f'epoch_{epoch+1}')
    os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_save.save_pretrained(save_directory)

