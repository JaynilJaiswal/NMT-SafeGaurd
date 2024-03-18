import torch
from model import CustomMTModel
from utils import *
from torch.utils.data import DataLoader
import pickle


# Load data from pickle file
with open('all_inputs_250', 'rb') as f:
    data = pickle.load(f)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize custom dataset and dataloader
custom_dataset = CustomDataset(data)
dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=True, collate_fn=custom_dataset.collate_func)

# Initialize model
model = CustomMTModel(device)
model.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def compute_loss(outputs, targets):
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    # Assuming targets are labels for language modeling (next word prediction)
    loss = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for language modeling
    return loss(logits.view(-1, logits.size(-1)), targets.view(-1))

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for batch in dataloader:
        # Unpack batch
        # for x in batch:
        #     print(x.size())
        inputs_en, inputs_de, mask_en, mask_de = batch     

        
        # print(next(model.parameters()).device)
        # print(device)
        # Forward pass
        outputs = model(inputs_en.to(device), mask_en.to(device), inputs_de.to(device), mask_de.to(device))

        # Compute loss
        loss = compute_loss(outputs, inputs_de.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if batch % 10 == 9:
            print(f"Epoch [{epoch + 1}/{num_epochs}], batch {batch}, Loss: {loss.item():.4f}")
            # Save trained model
            torch.save(model.state_dict(), "model.pth")
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
