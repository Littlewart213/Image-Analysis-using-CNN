import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(101)

# Define transformations
transform = transforms.ToTensor()

# Download MNIST dataset
train_data = datasets.MNIST(root='./CNN/Data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./CNN/Data', train=False, download=True, transform=transform)

# Create data loaders (initial values)
train_batch_size = 64
test_batch_size = 64
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# Set batch sizes and number of epochs using Streamlit sidebar
train_batch_size = st.sidebar.slider("Train Batch Size", min_value=1, max_value=256, value=64, step=1)
test_batch_size = st.sidebar.slider("Test Batch Size", min_value=1, max_value=256, value=64, step=1)
epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=20, value=10, step=1)

# Create updated data loaders
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

# Define the Convolutional Neural Network model
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create an instance of the CNN model
model = ConvolutionalNeuralNetwork()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Lists to store training losses and accuracies
train_losses = []
train_accuracies = []

# Function to train the model
def train():
    train_loss_placeholder = st.empty()
    train_acc_placeholder = st.empty()
    epoch_placeholder = st.empty()
    batch_placeholder = st.empty()

    for epoch in range(epochs):
        train_loss = 0
        train_correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == target).sum().item()

            # Display batch loss and accuracy
            if batch_idx % 100 == 0:
                batch_loss = train_loss / (batch_idx + 1)
                batch_accuracy = 100.0 * train_correct / ((batch_idx + 1) * train_batch_size)
                batch_placeholder.write(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%")

                # Display example training data and labels
                if batch_idx == 0:
                    images = data[:5]
                    labels = target[:5]
                    fig, axs = plt.subplots(1, 5, figsize=(10, 3))
                    for i, ax in enumerate(axs):
                        ax.imshow(images[i].squeeze(), cmap='gray')
                        ax.set_title(f"Label: {labels[i].item()}")
                        ax.axis('off')
                    st.pyplot(fig)

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        train_loss_placeholder.write(f"\nEpoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # Calculate test loss and accuracy
    test_loss = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)

    st.write(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Confusion Matrix
    with torch.no_grad():
        predicted_labels = []
        true_labels = []
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(target.tolist())

    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    st.write("Confusion Matrix:")
    st.write(confusion_mat)

    # Plotting loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(train_accuracies, label='Training Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy')
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())

# Add a "Start" button
if st.button("Start"):
    # Train the model
    train()
