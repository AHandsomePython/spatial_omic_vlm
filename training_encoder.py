import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

print("1. Setting up the Custom Dataset...")
class TumorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# next step: organize your actual data loading here, replacing the dummy data below
# /data
#     /validation
#     /train
#         /tumor
#         ├── tissue_hires_image_lung_1_tumor.jpg
#         ├── tissue_hires_image_lung_2_tumor.jpg
#         ...

# --- DUMMY DATA (Replace with the actual folder parsing) ---
sample_images = ["./data/tissue_hires_image_lung_1_tumor.jpg", "./data/tissue_hires_image_lung_2_tumor.jpg"] 
sample_labels = [1, 1] 

dataset = TumorDataset(sample_images, sample_labels, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


print("2. Building the Model with a 'Temporary Head'...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

num_classes = 2 # Change this to match the actual number of classes
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet.to(device)


print("3. Setting up the Loss and Optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0001) 


print("4. Starting the Training Loop.\n")
epochs = 5

resnet.train()

for epoch in range(epochs):
    running_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = resnet(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Error (Loss): {running_loss/len(dataloader):.4f}")

resnet.fc = nn.Identity()