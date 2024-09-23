import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt


# Define the Lightning Module
class CIFAR10_ResNet18(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(CIFAR10_ResNet18, self).__init__()
        # Initialize the ResNet-18 model
        self.model = models.resnet18(pretrained=False, num_classes=10)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        # Initialize lists to store accuracies for plotting
        self.train_acc_history = []
        self.val_acc_history = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.train_acc_history.append(acc)

        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Calculate validation accuracy
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean()
        self.val_acc_history.append(acc)

        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Instantiate the model
model = CIFAR10_ResNet18()

# Train the model using PyTorch Lightning Trainer
trainer = Trainer(max_epochs=20, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
trainer.fit(model, train_loader, val_loader)

# Plot accuracy curves after training
plt.figure(figsize=(10, 5))
plt.plot(model.train_acc_history, label='Training Accuracy')
plt.plot(model.val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.legend()
plt.show()
