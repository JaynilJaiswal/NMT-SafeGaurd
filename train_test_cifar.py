import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np 
from main import evaluate_adversarial_examples
from c import test_generator
import torch
from torch.utils.data import TensorDataset, DataLoader



#################### DEFINE MODELS AND CUSTOM LOSS ####################
# class Generator(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(Generator, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 3072)  # Output size is the same as input size (MNIST images)
#         self.activation = nn.Tanh()

#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return x

class Generator(nn.Module):
    def __init__(self, channels_img=3, features_g=64):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            # Input: N x channels_img x 32 x 32
            nn.Conv2d(channels_img, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),

            # Input: N x features_g x 16 x 16
            nn.Conv2d(features_g, features_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(),

            # Input: N x features_g*2 x 8 x 8
            nn.Conv2d(features_g*2, features_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(),

            # Input: N x features_g*4 x 4 x 4
            nn.Conv2d(features_g*4, features_g*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*8),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # Input: N x features_g*8 x 2 x 2
            nn.ConvTranspose2d(features_g*8, features_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*4),
            nn.ReLU(),

            # Input: N x features_g*4 x 4 x 4
            nn.ConvTranspose2d(features_g*4, features_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g*2),
            nn.ReLU(),

            # Input: N x features_g*2 x 8 x 8
            nn.ConvTranspose2d(features_g*2, features_g, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g),
            nn.ReLU(),

            # Input: N x features_g x 16 x 16
            nn.ConvTranspose2d(features_g, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range: [-1,1]
            # Output: N x channels_img x 32 x 32
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    
class CifarClassifier(nn.Module):
    def __init__(self):
        super(CifarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target):
        logs = torch.log(pred)
        for i in range(target.shape[0]):
            logs[i, target[i].item()] = 0
        return -torch.sum(logs, dim=1) / (pred.shape[1])
    

#################### LOAD DATASET ####################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(cifar_dataset, batch_size=128, shuffle=True)

# creating a dataset of adversarial examples and then using this dataset to initialize a DataLoader:
def create_adversarial_data_loader(generator, original_data_loader, device):
    generator.eval()
    adversarial_images = []
    labels = []

    with torch.no_grad():
        for data, target in original_data_loader:
            data = data.to(device)
            adversarial_data = generator(data)
            adversarial_images.append(adversarial_data.cpu())
            labels.append(target)

    # Concatenate all the adversarial images and labels
    adversarial_images = torch.cat(adversarial_images, 0)
    labels = torch.cat(labels, 0)

    # Create a TensorDataset with adversarial images
    adversarial_dataset = TensorDataset(adversarial_images, labels)

    # Create a DataLoader from the adversarial dataset
    adversarial_data_loader = DataLoader(adversarial_dataset, batch_size=original_data_loader.batch_size, shuffle=False)

    return adversarial_data_loader

#################### LOAD MODELS ####################
generator = Generator()
classifier = CifarClassifier()

def load_model(model, model_name):
    '''Check if pre-trained models exist and load them'''
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Pre-trained model '{model_name}' loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"No pre-trained model found. Training from scratch.")
        return False

generator_trained = load_model(generator, 'generator_cifar.pth')
classifier_trained = load_model(classifier, 'classifier_cifar.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
classifier.to(device)


#################### PLOT IMAGE FUNCTION ####################
def plot_images(real_images, generated_images, epoch):
    print(real_images.size())
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 4))
    for ax, image in zip(axes[0], real_images):
        print(image.size())
        ax.imshow(np.transpose(image.cpu().detach().numpy(), (1,2,0)))
        ax.axis('off')
    for ax, image in zip(axes[1], generated_images):
        ax.imshow(np.transpose(image.cpu().detach().numpy(), (1,2,0)))
        ax.axis('off')
    plt.savefig(f'comparison_epoch_{epoch}_cifar.png')
    plt.close()



################### TRAIN CLASSIFIER IF NOT LOADED ####################
if not classifier_trained:
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # images = images.view(-1, 3072).to(device)
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = correct_predictions / total_predictions * 100
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    print('Training completed.')
    torch.save(classifier.state_dict(), 'classifier_cifar.pth')


print("Generator :-")
num_epochs_gen = 50
#################### TRAIN GENERATOR IF NOT LOADED ####################
if not generator_trained:
    classifier.eval()
    criterion = AdversarialLoss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    num_epochs = 10

    for epoch in range(num_epochs_gen):
        gen_loss_sum = 0.0
        disc_loss_sum = 0.0
        pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in pbar:
            
            images = images.to(device)
            # images = images.to(device)
            generator.zero_grad()
            generated_images = generator(images)
            cosine_similarity = F.cosine_similarity(generated_images, images, dim=1).mean() # Cosine similarity loss
            generated_images = generated_images.view(-1,3,32,32)
            adversarial_loss = criterion(classifier(generated_images), labels)              # Adversarial loss from classifier

            # Total generator loss
            generator_loss = (1-cosine_similarity) + 0.05*torch.mean(adversarial_loss)
            generator_loss.backward()
            gen_optimizer.step()

            gen_loss_sum += generator_loss.item()
            pbar.set_postfix({'Generator Loss': gen_loss_sum / len(data_loader)})
        
        if epoch % 1 == 0:
            images = images.view(-1,3,32,32)
            generated_images = generated_images.view(-1,3,32,32)
            plot_images(images[:10], generated_images[:10], epoch)
    torch.save(generator.state_dict(), 'generator_cifar.pth')
    print("Training completed.")

# benchmark for: Classification Accuracy Drop
# Evaluate the classifier on the original dataset
original_accuracy = test_generator(classifier, data_loader, device)
# Evaluate the classifier on adversarial examples
adversarial_data_loader = create_adversarial_data_loader(generator, data_loader, device)
adversarial_accuracy = evaluate_adversarial_examples(generator, classifier, adversarial_data_loader, device)

# Calculate the accuracy drop
accuracy_drop = original_accuracy - adversarial_accuracy
print(f'Original Accuracy: {original_accuracy:.2f}%')
print(f'Adversarial Accuracy: {adversarial_accuracy:.2f}%')
print(f'Accuracy Drop: {accuracy_drop:.2f}%')

# noise_dim = 100


# # Testing accuracy using generated images
# def test_classifier_accuracy(generator, classifier, num_samples=10):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     generator.to(device)
#     classifier.to(device)
#     # Define data transformations
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     # Load MNIST test dataset
#     mnist_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#     # Define DataLoader for the test dataset
#     test_loader = DataLoader(mnist_test_dataset, batch_size=1, shuffle=False)

#     generator.eval()
#     classifier.eval()

#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, (real_image, label) in enumerate(test_loader):
#             # Generate random noise and a real image
#             # noise = torch.randn(1, noise_dim, device=device)
#             # real_image = torch.randn(1, 784, device=device)  # Assuming random noise as input, you may change this
#             # Take one real image from the dataset
#             # print(label)
#             real_image = real_image.view(-1, 784).to(device)
            
#             # print(real_image.size())
#             # Generate a fake image using the generator
#             # fake_image = generator(torch.cat([noise, real_image], dim=1))
#             fake_image = generator(real_image)
#             #plot_image(real_image, fake_image, batch_idx)
#             # Pass the fake image through the classifier
#             outputs = classifier(real_image)
#             _, predicted = torch.max(outputs.data, 1)
           
#             # Increment counters
#             total += 1
#             if predicted.item() == label.item():  # Check if the predicted label is the same as the generated image label
#                 correct += 1

#     accuracy = correct / total * 100
#     print(f"Accuracy of the classifier using generated images: {accuracy:.2f}%")

# # Test classifier accuracy using generated images
# test_classifier_accuracy(generator, classifier)