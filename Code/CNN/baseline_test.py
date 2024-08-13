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
n_classes = 8
height = 120
width = 160
bs = 32

# Load multiple templates for each class
templates = {
    'platform': [cv2.imread(f'baseline_model/Objects/1_platform/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 4)],
    'spike': [cv2.imread(f'baseline_model/Objects/2_spike/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)],
    'player': [cv2.imread(f'baseline_model/Objects/player_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)],
    'yellow_jump_orb': [cv2.imread(f'baseline_model/Objects/yellow_jump_orb_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)],
    'blue_jump_orb': [cv2.imread(f'baseline_model/Objects/blue_jump_orb_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)],
    'blue_pad': [cv2.imread(f'baseline_model/Objects/blue_pad_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)],
    'yellow_pad': [cv2.imread(f'baseline_model/Objects/yellow_pad_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)],
    'portal': [cv2.imread(f'baseline_model/Objects/portal_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 3)]
}

# Define a uniform threshold for all classes
thresholds = [0.7] * n_classes

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor(),
])

# Define paths for images and annotations
image_path_dir = 'Sample_Data/polargeist/polargeist_normal'
annot_path_dir = 'Sample_Data/polargeist/polargeist_hitbox'
image_paths = []
annot_paths = []
for i in range(1, 511):
    image_paths.append(image_path_dir + '/' + str(i) + '.png')
    annot_paths.append(annot_path_dir + '/' + str(i) + '.png')

# Create dataset and split into train, validation, and test sets
dataset = CustomDataset(image_paths=image_paths,
                        annot_paths=annot_paths,
                        transform=transform)

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

# Process each image in the test loader
total_pixels = 0
total_diff_pixels = 0

for images, masks in test_loader:
    for i in range(images.shape[0]):  # Iterate over the batch
        img = images[i].numpy().squeeze() * 255  # Convert image to grayscale
        img = img.astype(np.uint8)

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
        diff = output_tensor - expected_tensor
        num_diff_pixels = np.sum(diff != 0)
        num_total_pixels = expected_tensor.size

        total_diff_pixels += num_diff_pixels
        total_pixels += num_total_pixels

        print(f'Number of different pixels for image {i}: {num_diff_pixels}')

# Calculate accuracy
accuracy = 1 - (total_diff_pixels / total_pixels)
print(f'Overall accuracy: {accuracy * 100:.2f}%')

# Optional: Visualize the result for one of the layers (e.g., platform layer)
cv2.imwrite('platform_layer.png', output_tensor[0] * 255)
