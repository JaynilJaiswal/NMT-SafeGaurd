import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.activation(x)
        return x

# Noise dimension for generator input
noise_dim = 100

# Define the generator network
generator = Generator(input_size=784, hidden_size=256)  # Input size = 784 (image) + 10 (conditional noise)
generator.load_state_dict(torch.load('generator.pth'))


# Load pretrained discriminator
pretrained_discriminator = Discriminator(input_size=784, hidden_size=256)
pretrained_discriminator.load_state_dict(torch.load('classifier.pth'))

# Test the generator
def test_generator(generator, pretrained_discriminator, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    pretrained_discriminator.to(device)
    
    generator.eval()
    pretrained_discriminator.eval()

    with torch.no_grad():
        for _ in range(num_samples):
            # Generate random noise and a real image
            # noise = torch.randn(1, noise_dim, device=device)
            real_image = torch.randn(1, 784, device=device)  # Assuming random noise as input, you may change this
            
            # Generate a fake image using the generator
            # fake_image = generator(torch.cat([noise, real_image], dim=1))
            fake_image = generator(real_image)
            
            # Pass the fake image through the pretrained discriminator
            discriminator_output = pretrained_discriminator(fake_image)
            
            # Print the discriminator output
            print("Discriminator Output:", discriminator_output.item())

# Test the generator
test_generator(generator, pretrained_discriminator)
