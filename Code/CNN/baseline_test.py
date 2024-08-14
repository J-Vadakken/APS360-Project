import os
import cv2 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt 
import torch.optim as optim
from Custom_Data_loader import CustomDataset
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image

# Define the number of classes and initialize the output tensor dimensions
n_classes = 10
height = 270
width = 480
bs = 32
# Load and resize multiple templates for each class to 3/4 of their original size
templates = {
    '0': [cv2.resize(cv2.imread(f'baseline_model/Objects/0/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/0/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/0/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(11)],
    '1': [cv2.resize(cv2.imread(f'baseline_model/Objects/1/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/1/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/1/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(11)],
    '2': [cv2.imread(f'baseline_model/Objects/2/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 4)],
    '3': [cv2.resize(cv2.imread(f'baseline_model/Objects/3/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/3/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/3/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(1, 2)],
    '4': [cv2.resize(cv2.imread(f'baseline_model/Objects/4/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/4/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/4/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(3, 4)],
    '5': [cv2.resize(cv2.imread(f'baseline_model/Objects/5/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/5/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/5/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(6, 7)],
    '6': [cv2.resize(cv2.imread(f'baseline_model/Objects/6/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/6/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/6/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(4, 5)],
    '7': [cv2.resize(cv2.imread(f'baseline_model/Objects/7/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/7/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/7/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(2, 3)],
    '8': [cv2.resize(cv2.imread(f'baseline_model/Objects/8/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/8/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/8/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(5, 6)],
    '9': [cv2.resize(cv2.imread(f'baseline_model/Objects/9/{i}.png', cv2.IMREAD_GRAYSCALE), 
                     (int(cv2.imread(f'baseline_model/Objects/9/{i}.png', cv2.IMREAD_GRAYSCALE).shape[1] * 3 / 4), 
                      int(cv2.imread(f'baseline_model/Objects/9/{i}.png', cv2.IMREAD_GRAYSCALE).shape[0] * 3 / 4))) for i in range(1, 6)],
}

# Example check of the resized template's shape
print(templates['0'][0].shape)



# Define a uniform threshold for all classes
thresholds = [0.7] * n_classes

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((270, 480)),
    transforms.ToTensor(),
])

# Initialize lists to hold all image and annotation paths
all_image_paths = []
all_annot_paths = []

# Define the root directory
root_dir = 'Sample_Data/training_validation_data'

# Loop through each directory in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    if os.path.isdir(subdir_path):
        # Define paths for images and annotations for each level
        image_path_dir = os.path.join(subdir_path, f'{subdir}_Normal')
        annot_path_dir = os.path.join(subdir_path, f'{subdir}_Hitbox')

        # Ensure the directories exist
        if os.path.exists(image_path_dir) and os.path.exists(annot_path_dir):
            # List the files in each directory and append to the overall list
            image_paths = [os.path.join(image_path_dir, f) for f in os.listdir(image_path_dir) if f.endswith('.png')]
            annot_paths = [os.path.join(annot_path_dir, f) for f in os.listdir(annot_path_dir) if f.endswith('.png')]
            
            # Extend the overall lists with these paths
            all_image_paths.extend(image_paths)
            all_annot_paths.extend(annot_paths)

# Create dataset and split into train, validation, and test sets
dataset = CustomDataset(image_paths=all_image_paths,
                        annot_paths=all_annot_paths,
                        transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# Process each image in the test loader
total_pixels = 0
total_diff_pixels = 0
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize lists to store metrics for each class
precision_per_class = []
recall_per_class = []
f1_per_class = []

# Initialize arrays to accumulate true positives, false positives, and false negatives
tp = np.zeros(n_classes, dtype=int)
fp = np.zeros(n_classes, dtype=int)
fn = np.zeros(n_classes, dtype=int)
for images, masks in test_loader:
    for i in range(images.shape[0]):  # Iterate over the batch
        img = images[i].numpy().squeeze() * 255  # Convert image to grayscale
        img = img.astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))  # Reorder to (height, width, channels)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output_tensor = np.zeros((n_classes, height, width), dtype=np.uint8)  # Reset output tensor

        for idx, (key, template_list) in enumerate(templates.items()):
            for template in template_list:
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= thresholds[idx])
                w, h = template.shape[::-1]
                for pt in zip(*loc[::-1]):
                    output_tensor[idx, pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 1

        # Compare the output tensor with the expected tensor
        expected_tensor = masks[i].numpy()

        for j in range(n_classes):
            tp[j] += np.sum((output_tensor[j] == 1) & (expected_tensor[j] == 1))
            fp[j] += np.sum((output_tensor[j] == 1) & (expected_tensor[j] == 0))
            fn[j] += np.sum((output_tensor[j] == 0) & (expected_tensor[j] == 1))

# Calculate precision, recall, and F1 score for each class
for j in range(n_classes):
    precision = tp[j] / (tp[j] + fp[j]) if tp[j] + fp[j] > 0 else 0
    recall = tp[j] / (tp[j] + fn[j]) if tp[j] + fn[j] > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    precision_per_class.append(precision)
    recall_per_class.append(recall)
    f1_per_class.append(f1)

    print(f'Class {j}: Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# Optional: Calculate overall metrics (macro-average)
overall_precision = np.mean(precision_per_class)
overall_recall = np.mean(recall_per_class)
overall_f1 = np.mean(f1_per_class)

print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')
print(f'Overall F1 Score: {overall_f1:.4f}')