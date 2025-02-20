import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from timm import create_model
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import torch
import time
#from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast


print("Libraries resolved", flush=True)
# Paths to dataset files
# Provide Valid path to the file
train_val_list_path = 'Valid path to the file /train_val_list.txt'
test_list_path = 'Valid path to the file /test_list.txt'
data_entry_path = 'Valid path to the file /Data_Entry_2017.csv'
image_dir = 'Valid path to the file /images'
checkpoint_path = "Valid path to the file "


def get_full_image_path(filename):
    return os.path.join(image_dir, filename)

def load_image_paths(txt_file_path):
    with open(txt_file_path, 'r') as file:
        filenames = file.readlines()
    return [get_full_image_path(line.strip()) for line in filenames]



def load_image_labels(csv_file_path, image_paths):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)
    print("Full data",df.shape, flush=True)
    df = df.dropna(subset=['Finding Labels'])
    print("Drop na",df.shape, flush=True)
    # Create a dictionary for fast lookup
    label_dict = dict(zip(df['Image Index'], df['Finding Labels']))
    df = df[df['Finding Labels'].str.strip() != ""]
    print("Drop ''",df.shape, flush=True)

    # Prepare the image labels based on the image paths
    image_labels = []
    for image_path in image_paths:
        image_filename = os.path.basename(image_path)

        # Fetch the labels directly from the dictionary
        label = label_dict.get(image_filename, "")
        labels_split = label.split('|') if label else []  # Split labels by '|' or return empty if not found
        image_labels.append(labels_split)  # Store as a list of labels

    return image_labels

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and preprocess the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)  # Ensure label is float32 for BCEWithLogitsLoss
        return image, label






# Define the training function with improvements
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, scheduler=None):
    # Mixed precision scaler
    scaler = GradScaler(device='cuda')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels.float())

            # Backward pass with scaled loss
            scaler.scale(loss).backward()

            # Update weights with optimizer
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{total_batches}], "
                      f"Batch Loss: {loss.item():.4f}", flush=True)

        epoch_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed, Loss: {epoch_loss:.4f}, '
              f'Time Taken: {time.time() - start_time:.2f} seconds', flush=True)

        # Adjust learning rate using scheduler if available
        if scheduler:
            scheduler.step(epoch_loss)

        # Validate the model after each epoch
        print("Validating model...")
        # Validate the model after each epoch
        print("Validating model...", flush=True)
        val_loss, val_accuracy = evaluate_model(model, val_loader, device, criterion)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n', flush=True)
        # Save the model
        torch.save({'model_state_dict':model.state_dict()}, 'convnext_model.pth')

# Enhanced evaluate_model function to compute and print AUC, Precision, Recall, etc.
def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    total, correct_subsets = 0, 0
    running_val_loss = 0.0
    total_batches = len(test_loader)
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Threshold for multi-label classification

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # If a loss function is provided, compute the validation loss
            if criterion:
                loss = criterion(outputs, labels.float())
                running_val_loss += loss.item()

            total += labels.size(0)
            correct_subsets += (predicted == labels).all(dim=1).sum().item()
    
    val_loss = running_val_loss / total_batches if criterion else 0
    subset_accuracy = 100 * correct_subsets / total_batches

    # Convert all_labels and all_predictions to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Initialize lists to store per-class AUC
    auc_per_class = []

    # Calculate AUC for each class
    for i in range(all_labels.shape[1]):
        y_true = all_labels[:, i]
        y_pred = all_predictions[:, i]

        # Check if we have both 0 and 1 in y_true for this class
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            auc_per_class.append(auc)
        else:
            auc_per_class.append(None)  # Skip AUC for this class as only one label is present

    # Calculate other metrics
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    # Print metrics
    print(f'Validation Loss: {val_loss:.4f}', flush=True)
    print(f'Accuracy: {subset_accuracy:.2f}%', flush=True)
    print(f'Precision: {precision:.4f}', flush=True)
    print(f'Recall: {recall:.4f}', flush=True)
    print(f'F1 Score: {f1:.4f}', flush=True)

    # Print AUC for each class
    for i, auc in enumerate(auc_per_class):
        if auc is not None:
            print(f'AUC for class {i}: {auc:.4f}', flush=True)
        else:
            print(f'AUC for class {i}: Not defined (only one class present)', flush=True)

    return val_loss, subset_accuracy


# Load image paths
print("starting....", flush=True)
train_val_image_paths = load_image_paths(train_val_list_path)
test_image_paths = load_image_paths(test_list_path)
print(len(test_image_paths))
# Load image labels
train_val_labels = load_image_labels(data_entry_path, train_val_image_paths)
test_labels = load_image_labels(data_entry_path, test_image_paths)

# Convert labels to multi-hot encoding using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_val_labels = mlb.fit_transform(train_val_labels)
test_labels = mlb.transform(test_labels)
num_classes = len(mlb.classes_)
print(f"Number of classes: {num_classes}", flush=True)

# Split the training/validation data
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_image_paths, train_val_labels, test_size=0.2, random_state=42)

# Define any necessary transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images
    transforms.ToTensor(),           # Convert PIL images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
])


# Create the datasets
train_dataset = ImageDataset(train_paths, train_labels, transform=transform)
val_dataset = ImageDataset(val_paths, val_labels, transform=transform)
test_dataset = ImageDataset(test_image_paths, test_labels, transform=transform)

# Load the pre-trained ConvNeXt model
model = create_model('convnext_base', pretrained=True, num_classes=num_classes)

# Send model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)
model = model.to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Check if checkpoint exists before loading
if os.path.exists(checkpoint_path):
    print("Checkpoint found! Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    
    # Load the model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
else:
    print("No checkpoint found, starting training from scratch.", flush=True)






print("...loading datasets...", flush=True)



# Train the model
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("loading datasets complete tarining started...", flush=True)

train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)


# Evaluate the model
test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)

# Print the test metrics
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%', flush=True)
# Save the model
torch.save({'model_state_dict':model.state_dict()}, 'convnext_model.pth')

