import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from config import imshape_, n_classes_, labels_, class_colors_
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, image_paths, annot_paths, transform=None):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annot_path = self.annot_paths[idx]

        # Load image and annotation
        image = Image.open(image_path).convert("RGB")
        annot_image = Image.open(annot_path).convert("RGB")

        # Apply transformations (Ensure no image augmentation is applied to the annotation image)
        to_tensor_transform = transforms.ToTensor()
        if self.transform:
            image = self.transform(image)
            annot_image = self.transform(annot_image)
        else:
            image = to_tensor_transform(image)
            annot_image = to_tensor_transform(annot_image)

        # Generate masks
        annot_image = annot_image * 255.0
        mask = self.create_multi_masks(annot_image)

        return image, mask

    def create_multi_masks(self, anot_im):
        # Convert anot_im to numpy if it's a tensor
        if torch.is_tensor(anot_im):
            anot_im = anot_im.numpy()
            # Assuming anot_im is in CHW format, convert it to HWC for processing
            anot_im = anot_im.transpose(1, 2, 0)
        
        background_mask = np.ones((anot_im.shape[0], anot_im.shape[1]), dtype=np.float32)
        channels = []

        plt.imshow(anot_im)
        plt.show()
        plt.figure(figsize=(10, n_classes_*2.5))
        i = 1
        for label, color in class_colors_.items():
            color = np.array(color, dtype=anot_im.dtype)
            tolerance = 1
            mask = np.all(np.abs(anot_im - color) <= tolerance, axis=-1)
            background_mask[mask] = 0
            mask = mask.astype(np.float32)
            channels.append(mask)
            plt.subplot(n_classes_, 2, 2*i-1)
            plt.imshow(mask, cmap='gray')
            plt.title(label)
            plt.subplot(n_classes_, 2, 2*i)
            plt.imshow(background_mask, cmap='gray')
            plt.title(color)
            i+= 1
        plt.subplots_adjust(hspace=1, wspace=0.5)
        plt.show()


        channels.append(background_mask)
        y = np.stack(channels, axis=0)
        return torch.tensor(y, dtype=torch.float32)

# Test code:
if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(image_paths=['Code/CNN/Sample_Data/polargeist/polargeist normal/2024 Jul 02 22-08-07241.png'],
                            annot_paths=['Code/CNN/Sample_Data/polargeist/polargeist hitbox/2024 Jul 02 21-44-574.png'],
                            )
    #     dataset = CustomDataset(image_paths=['Code/CNN/Sample_Data/polargeist/polargeist hitbox/2024 Jul 02 21-44-574.png'],
    #                        annot_paths=['Code/CNN/Sample_Data/polargeist/polargeist normal/2024 Jul 02 22-08-07241.png'],
    #                       transform=transform)
    a_img = Image.open('Code/CNN/Sample_Data/polargeist/polargeist hitbox/2024 Jul 02 21-44-574.png').convert("RGB")
    to_tensor_transform = transforms.ToTensor()
    a_img = to_tensor_transform(a_img) * 255.0
    f = dataset.create_multi_masks(a_img)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)