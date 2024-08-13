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
height = 120
width = 160
bs = 32

# Load multiple templates for each class
templates = {
    '0': [cv2.imread(f'baseline_model/Objects/0/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(11)],
    '1': [cv2.imread(f'baseline_model/Objects/1/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(11)],
    '2': [cv2.imread(f'baseline_model/Objects/2/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 4)],
    '3': [cv2.imread(f'baseline_model/Objects/3/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 2)],
    '4': [cv2.imread(f'baseline_model/Objects/4/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(3, 4)],
    '5': [cv2.imread(f'baseline_model/Objects/5/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(6, 7)],
    '6': [cv2.imread(f'baseline_model/Objects/6/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(4, 5)],
    '7': [cv2.imread(f'baseline_model/Objects/7/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(2, 3)],
    '8': [cv2.imread(f'baseline_model/Objects/8/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(5, 6)],
    '9': [cv2.imread(f'baseline_model/Objects/9/{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 6)],
}
print(templates['0'][0].shape)
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
        # make img greyscale
        print("img shape")
        print(img.shape)
        img = np.transpose(img, (1, 2, 0))  # Reorder to (height, width, channels)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        output_tensor = np.zeros((n_classes, height, width), dtype=np.uint8)  # Reset output tensor

        for idx, (key, template_list) in enumerate(templates.items()):
            for template in template_list:
                print(template.shape)
                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= thresholds[idx])
                w, h = template.shape[::-1]
                for pt in zip(*loc[::-1]):
                    output_tensor[idx, pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 1

        # Compare the output tensor with the expected tensor
        expected_tensor = masks[i].numpy()
        print(expected_tensor.shape)
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
cv2.imwrite('platform_layer.png', output_tensor[1] * 255)
