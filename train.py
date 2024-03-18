import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F


#################### DEFINE MODELS AND CUSTOM LOSS ####################
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
    
class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, pred, target):
        logs = torch.log(pred)
        for i in range(target.shape[0]):
            logs[i, target[i].item()] = 0
        return -torch.sum(logs, dim=1) / (pred.shape[1])
    

#################### LOAD DATASET ####################
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # Randomly rotate the image by ±10 degrees.
    transforms.RandomCrop(26, padding=4),  # Randomly crop the image and pad it to keep the size constant.
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
    transforms.Normalize((0.5,), (0.5,)),  # Normalize pixel values.
])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True)


#################### LOAD MODELS ####################
generator = Generator(input_size=784, hidden_size=256)
classifier = DigitClassifier(input_size=784, hidden_size=256)

def load_model(model, model_name):
    '''Check if pre-trained models exist and load them'''
    try:
        model.load_state_dict(torch.load(model_name))
        print(f"Pre-trained model '{model_name}' loaded successfully.")
        return True
    except FileNotFoundError:
        print(f"No pre-trained model found. Training from scratch.")
        return False

generator_trained = load_model(generator, 'generator.pth')
classifier_trained = load_model(classifier, 'classifier.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
classifier.to(device)


#################### PLOT IMAGE FUNCTION ####################
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
            images = images.view(-1, 784).to(device)
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
    torch.save(classifier.state_dict(), 'classifier.pth')


#################### TRAIN GENERATOR IF NOT LOADED ####################
if not generator_trained:
    classifier.eval()
    criterion = AdversarialLoss()
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    num_epochs = 10

    for epoch in range(num_epochs):
        gen_loss_sum = 0.0
        disc_loss_sum = 0.0
        pbar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in pbar:
            images = images.view(-1, 784).to(device)
            
            generator.zero_grad()
            generated_images = generator(images)

            cosine_similarity = F.cosine_similarity(generated_images, images, dim=1).mean() # Cosine similarity loss
            adversarial_loss = criterion(classifier(generated_images), labels)              # Adversarial loss from classifier

            # Total generator loss
            generator_loss = (1-cosine_similarity) + torch.mean(adversarial_loss)
            #print(torch.mean(adversarial_loss).item())
            generator_loss.backward()
            gen_optimizer.step()

            gen_loss_sum += generator_loss.item()
            pbar.set_postfix({'Generator Loss': gen_loss_sum / len(data_loader)})
        
        if epoch % 1 == 0:
            # Plot and save real and generated images every epoch
            plot_images(images[:10], generated_images[:10], epoch)
    torch.save(generator.state_dict(), 'generator.pth')
    print("Training completed.")