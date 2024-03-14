import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

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

# Define the generator and discriminator networks
generator = Generator(input_size=784, hidden_size=256)  # Input size = 784 (image) + 10 (conditional noise)
discriminator = Discriminator(input_size=784, hidden_size=256)

# Check if pre-trained models exist and load them
def load_model(model, model_name):
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Pre-trained model '{model_name}' loaded successfully.")
    except FileNotFoundError:
        print(f"No pre-trained model found. Training from scratch.")

load_model(generator, 'generator.pth')
load_model(discriminator, 'discriminator.pth')

# Define loss function and optimizers
criterion = nn.BCELoss()  # Binary cross-entropy loss
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)

# Training the GAN
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Freeze discriminator during training
freeze_discriminator = False

# Function to plot and save real and generated images
def plot_images(real_images, generated_images, epoch):
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for ax, image in zip(axes[0], real_images):
        ax.imshow(image.view(28, 28).cpu().detach().numpy(), cmap='gray')
        ax.axis('off')
    for ax, image in zip(axes[1], generated_images):
        ax.imshow(image.view(28, 28).cpu().detach().numpy(), cmap='gray')
        ax.axis('off')
    plt.savefig(f'comparison_epoch_{epoch}.png')
    plt.close()

for epoch in range(num_epochs):
    gen_loss_sum = 0.0
    disc_loss_sum = 0.0
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
    for real_images, real_labels in pbar:
        real_images = real_images.view(-1, 784).to(device)
        
        # Add noise to the real images
        noise = torch.randn_like(real_images) * 0.1  # Add Gaussian noise with std=0.1
        noisy_images = real_images + noise
        
        # Train discriminator
        if not freeze_discriminator:
            for _ in range(1):
                discriminator.zero_grad()
                real_output = discriminator(real_images)
                real_loss = criterion(real_output, torch.ones_like(real_output))
                
                noise = torch.randn_like(real_images, device=device)
                # fake_images = generator(torch.cat([noise, noisy_images], dim=1))
                fake_images = generator(real_images)
                fake_output = discriminator(fake_images.detach())
                fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
                
                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_optimizer.step()
                
                disc_loss_sum += discriminator_loss.item()

        # Train generator
        generator.zero_grad()
        noise = torch.randn_like(real_images, device=device)
        # generated_images = generator(torch.cat([noise, noisy_images], dim=1))
        generated_images = generator(real_images)
        
        # Cosine similarity loss
        cosine_similarity = F.cosine_similarity(generated_images, noisy_images, dim=1).mean()
        
        # Generator loss from discriminator
        generator_loss_disc = criterion(discriminator(generated_images), torch.ones_like(fake_output))
        
        # Total generator loss
        generator_loss = 1-cosine_similarity + generator_loss_disc
        generator_loss.backward()
        gen_optimizer.step()

        gen_loss_sum += generator_loss.item()

        pbar.set_postfix({'Generator Loss': gen_loss_sum / len(data_loader),
                        'Discriminator Loss': disc_loss_sum / len(data_loader)})
    
    if epoch % 2 == 0:
        # Plot and save real and generated images every epoch
        plot_images(real_images[:10], generated_images[:10], epoch)
        # Save models
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(discriminator.state_dict(), 'discriminator.pth')

print("Training completed.")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
