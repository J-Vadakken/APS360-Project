import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import torch.optim as optim
# Custom Data Loader pwd is ../Custom_Data_loader.py
from Custom_Data_loader import CustomDataset
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
import numpy as np

# __________________________________________________________________________
# Extract the processed data and labels
import os

transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
])

root_path_dir = 'Code/CNN/Sample_Data/training_validation_data'
image_paths = []
annot_paths = []

i = 0

for level in os.listdir(root_path_dir):
    level_dir = root_path_dir + '/' + level
    normal_dir = level_dir + '/' + os.listdir(level_dir)[1]
    hitbox_dir = level_dir + '/' + os.listdir(level_dir)[0]
    for image in os.listdir(normal_dir):
        # print(image)
        image_path = normal_dir + '/' + image
        image_paths.append(image_path)
        i += 1
    for annot in os.listdir(hitbox_dir):
        annot_path = hitbox_dir + '/' + annot
        annot_paths.append(annot_path)

print("Total images: ", i)

dataset = CustomDataset(image_paths=image_paths,
                        annot_paths=annot_paths,
                        transform=transform)

# Split the dataset 

train_size = int(0.7 * len(dataset))  # 70% for training
val_size = int(0.15 * len(dataset))   # 15% for validation
test_size = len(dataset) - train_size - val_size  # Remaining 15% for testing

print("Train size: ", train_size, "Validation size: ", val_size, "Test size: ", test_size)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
inputs, labels = next(iter(train_loader))


#______________________________________________________________________________
n_classes_ = 11
class BaseCNN(nn.Module):
    def __init__(self, n_classes):
        super(BaseCNN, self).__init__()
        self.name = "BaseCNN"
        
# Encoder part with only two convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Decoder part with a single upsampling and convolution
        self.upconv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv2(x))
        
        # Decoder
        x = self.upsample(x)
        x = F.relu(self.upconv1(x))
        x = self.final_conv(x)
        
        return x

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.name = "SimpleCNN"
        
        # Encoder part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Decoder part (no skip connections)
        self.upconv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        
        x = F.relu(self.conv4(x))
        
        # Decoder (no skip connections)
        x = self.upsample(x)
        x = F.relu(self.upconv1(x))
        
        x = self.upsample(x)
        x = F.relu(self.upconv2(x))
        
        x = self.upsample(x)
        x = F.relu(self.upconv3(x))
        x = self.final_conv(x)
        
        return x


path = "Code/CNN/saved_models/GD_model_BaseCNN_bs32_lr0.1_epoch6"
BaseMODEL = BaseCNN(n_classes_)
BaseMODEL.load_state_dict(torch.load(path))
BaseMODEL.eval()

#____________________________________________________________________________

pred = BaseMODEL(inputs)

x = F.softmax(pred, dim=1)
max_vals, _ = torch.max(x, dim=1, keepdim=True)
binary_mask = (x == max_vals).float()
binary_mask[0][10][0][0] = 0
print(binary_mask[11][0][0])
print(binary_mask[0], x.shape)
# print(labels[0])
# print(binary_mask.type(), labels[0].type())
# print(binary_mask[0].dtype, labels[0].dtype)
# print(binary_mask[0].shape, labels[0].shape)

# m = torch.ones((11,120,160)).float()
# m[4][0][0] = 0
# print(torch.max(labels))

# CustomDataset.plot_multi_mask(m)
# CustomDataset.plot_multi_mask(labels[0])
CustomDataset.plot_multi_mask(binary_mask[0])

k = torch.tensor([[1,0,3],[1,2,3],[2,3,4]]).float()
m = F.softmax(k, dim=0)
print(m)
