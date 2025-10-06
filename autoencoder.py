import torch
import torch.nn as nn 
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import ssl

# Certificate bypass
ssl._create_default_https_context = ssl._create_unverified_context

#device configuration

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# defining the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28*28,128),
                nn.ReLU(),
                nn.Linear(128, 64)
                )

        self.decoder = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 28*28),
                nn.Sigmoid()
                )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Preparing the data

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
        root = "./data", train=True, download=True, transform=transform)

test_dataset = datasets.MNIST(
        root = "./data", train=False, download=True, transform=transform)


train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False)

# Instantiate model, loss and optimizer

model = Autoencoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr = 1e-3)


# The training loop

num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        images, _ = data
        images = images.view(images.size(0), -1).to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training finished!")

# Set the model to evlauation mode

model.eval()

with torch.no_grad():
    data = next(iter(test_loader))
    images, _ = data
    images_flat = images.view(images.size(0), -1).to(device)

    recon_images = model(images_flat)

    images = images.view(-1, 28, 28)
    recon_images = recon_images.view(-1,28, 28).cpu() # Move to CPU for numpy/plotting

# Plot the original and reconstructed images
plt.figure(figsize=(10,4))

for i in range(10):
    ax = plt.subplot(2, 10, i+ 1)
    plt.imshow(images[i], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 4:
        ax.set_title("Original Images")

    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(recon_images[i], cmap = "gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 4:
        ax.set_title("Reconstructed Images")

plt.show()
