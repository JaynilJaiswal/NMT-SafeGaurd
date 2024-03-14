import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 784)  # Output size is the same as input size (MNIST images)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the Discriminator network (modified as a classifier)
class DigitClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)  # Output size is 10 for 10 classes (digits)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.activation(x)
        return x

# Noise dimension for generator input
noise_dim = 100

def load_model(model, model_name):
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Pre-trained model '{model_name}' loaded successfully.")
    except FileNotFoundError:
        print(f"No pre-trained model found. Training from scratch.")


# Define the generator network
generator = Generator(input_size=784, hidden_size=256)  # Input size = 784 (image) + 10 (conditional noise)
load_model(generator, 'generator.pth')

# Define the classifier network
classifier = DigitClassifier(input_size=784, hidden_size=256)
load_model(classifier,"classifier.pth")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)

# Define loss function and optimizers
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train the classifier
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier.to(device)

# for epoch in range(num_epochs):
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0

#     for images, labels in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
#         images = images.view(-1, 784).to(device)
#         labels = labels.to(device)

#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass
#         outputs = classifier(images)

#         # Calculate loss
#         loss = criterion(outputs, labels)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         # Track the accuracy and loss
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total_predictions += labels.size(0)
#         correct_predictions += (predicted == labels).sum().item()

#     epoch_loss = running_loss / len(data_loader)
#     epoch_accuracy = correct_predictions / total_predictions * 100

#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# print('Training completed.')
# torch.save(classifier.state_dict(), 'classifier.pth')


# Testing accuracy using generated images
def test_classifier_accuracy(generator, classifier, num_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    classifier.to(device)
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST test dataset
    mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Define DataLoader for the test dataset
    test_loader = DataLoader(mnist_test_dataset, batch_size=1, shuffle=False)

    generator.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random noise and a real image
            # noise = torch.randn(1, noise_dim, device=device)
            # real_image = torch.randn(1, 784, device=device)  # Assuming random noise as input, you may change this
            real_image, label = next(iter(test_loader))  # Take one real image from the dataset
            # print(label)
            real_image = real_image.view(-1, 784).to(device)
            # print(real_image.size())
            # Generate a fake image using the generator
            # fake_image = generator(torch.cat([noise, real_image], dim=1))
            fake_image = generator(real_image)
            
            # Pass the fake image through the classifier
            outputs = classifier(fake_image)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            # Increment counters
            total += 1
            if predicted.item() == label.item():  # Check if the predicted label is the same as the generated image label
                correct += 1

    accuracy = correct / total * 100
    print(f"Accuracy of the classifier using generated images: {accuracy:.2f}%")

# Test classifier accuracy using generated images
test_classifier_accuracy(generator, classifier)
