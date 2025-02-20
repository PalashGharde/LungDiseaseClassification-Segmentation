import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np

# Custom dataset class
class LungSegmentationDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        org_img_path, label_img_path = self.file_list[idx].strip().split(',')
        # Replace backslashes with forward slashes for compatibility with Colab
        org_img_path = org_img_path.replace('\\', '/')
        label_img_path = label_img_path.replace('\\', '/')
        
        # Remove leading ".\" and create full paths
        org_img_path = os.path.join('Valid path to the file /Segmentation01', org_img_path[2:])
        label_img_path = os.path.join('Valid path to the file /Segmentation01', label_img_path[2:])
        
        # Load images
        org_img = cv2.imread(org_img_path, cv2.IMREAD_GRAYSCALE)
        label_img = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded correctly
        if org_img is None:
            raise FileNotFoundError(f"Original image not found: {org_img_path}")
        if label_img is None:
            raise FileNotFoundError(f"Label image not found: {label_img_path}")
        
        # Binarize the label
        label_img = np.where(label_img == 255, 1, 0)
        
        if self.transform:
            org_img = self.transform(org_img)
        
        return org_img, torch.tensor(label_img, dtype=torch.float32)

# Function to read the training list file
def load_file_list(filepath):
    with open(filepath, 'r') as file:
        return file.readlines()

# Load file list
train_file_list = load_file_list('Valid path to the file /list_train.txt')

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust for 3 channels
])

# Load the dataset using the parsed paths
train_dataset = LungSegmentationDataset(train_file_list, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Test dataset paths
test_file_list = load_file_list('Valid path to the file /list_test.txt')  # Assume a similar format for testing
test_dataset = LungSegmentationDataset(test_file_list, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Your model training and testing code would follow here


# Load ResNet-50 pretrained model and modify for segmentation
class ResNet50_Segmentation(nn.Module):
    def __init__(self):
        super(ResNet50_Segmentation, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        # Remove the fully connected layer and the avgpool layer
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        # Add a ConvTranspose2d layer for upsampling (decoder)
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # Final layer to output 1-channel mask
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = self.upconv(x)
        return x

# Instantiate the model
resnet50_segmentation = ResNet50_Segmentation()

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(resnet50_segmentation.parameters(), lr=0.001)
print("Training Started")
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    resnet50_segmentation.train()
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet50_segmentation(images)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(resnet50_segmentation.state_dict(), 'lung_segmentation_resnet50.pth')
