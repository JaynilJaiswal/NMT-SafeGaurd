import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model
from model import CifarClassifier

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def denorm(batch):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def test( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data).to(device)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)
        # Check for success
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 10:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

if __name__ == "__main__":

    epsilons = [0, .05, .1, .15, .2, .25, .3]
    classifier = CifarClassifier()
    load_model(classifier, 'classifier_cifar.pth')
    use_cuda=True
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    classifier.to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    # Load CIFAR test dataset
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    # Define DataLoader for the test dataset
    test_loader = DataLoader(cifar_dataset, batch_size=1, shuffle=False)

    class_labels = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(classifier, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    # Save the figure
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig, adv, ex = examples[i][j]
            # plt.title(f"{orig} -> {adv}")
            plt.title(f"{class_labels[orig]} -> {class_labels[adv]}") 
            ex_image = ex.reshape(3, 32, 32).transpose(1, 2, 0)
            plt.imshow(ex_image)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('adversarial_examples.png')
    plt.show()