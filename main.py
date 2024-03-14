import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import Autoencoder
from utils import WMT14Dataset
from datasets import load_dataset




# Load data from pickle file
# with open('all_inputs_250', 'rb') as f:
#     data = pickle.load(f)
# Load WMT14 dataset from Hugging Face
dataset = load_dataset("wmt14", "de-en")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define training function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        print(batch[1].size())
        input_ids = batch[0].squeeze().to(device)
        # attention_mask = input_ids != tokenizer.pad_token_id
        attention_mask = batch[1].squeeze().to(device)
        optimizer.zero_grad()
        reconstructed_logits = model(input_ids, attention_mask)
        print(reconstructed_logits.size(), input_ids.size())
        loss = criterion(reconstructed_logits.view(-1, reconstructed_logits.size(-1)), input_ids)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize dataset and dataloader
wmt14_dataset = WMT14Dataset(dataset, tokenizer=tokenizer)
dataloader = DataLoader(wmt14_dataset, batch_size=16, shuffle=True, collate_fn=wmt14_dataset.collate_fn)

# Initialize model, optimizer, and criterion
autoencoder = Autoencoder("bert-base-uncased")
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
autoencoder.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    loss = train(autoencoder, dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")