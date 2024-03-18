import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import Autoencoder
from utils import WMT14Dataset
from datasets import load_dataset
from project_loss import CustomLoss




# Load data from pickle file
# with open('all_inputs_250', 'rb') as f:
#     data = pickle.load(f)
# Load WMT14 dataset from Hugging Face
dataset = load_dataset("wmt14", "de-en", trust_remote_code=True)

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

# function that both generates adversarial examples and evaluates the classifier:
def evaluate_adversarial_examples(generator, classifier, data_loader, device):
    generator.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Generate adversarial examples
            adversarial_images = generator(images)

            # Evaluate classifier on adversarial examples
            outputs = classifier(adversarial_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the classifier on adversarial examples: {accuracy:.2f}%')
    return accuracy