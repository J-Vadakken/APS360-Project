import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import n_classes_, class_colors_, imshape_
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """
    Custom dataset class for loading images and annotations together.
    Creates multi-channel masks for each image annotation..
        Args:
        image_paths (list): List of paths to the images.
        annot_paths (list): List of paths to the annotations.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        Methods:
        __len__: Returns the length of the dataset.
        __getitem__: Returns a sample from the dataset at the given index.
        create_multi_masks: Returns multi-channel masks for the given annotation image.
        plot_multi_mask: Plots the multi-channel masks, given a mask tensor as an argument.
    """

    def __init__(self, image_paths, annot_paths, batch_size = 32, transform=None):
        self.image_paths = image_paths
        self.annot_paths = annot_paths
        self.batch_size = batch_size
        self.transform = transform
        self.indexes = np.arange(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Generate indexes of the batch
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        image_paths = [self.image_paths[k] for k in indexes]
        annot_paths = [self.annot_paths[k] for k in indexes]

        X, y = self.__data_generation(image_paths, annot_paths)

        return X, y
    
    def __data_generation(self, image_paths, annot_paths):
        
        X = np.empty((self.batch_size, imshape_[0], imshape_[1], imshape_[2]), dtype=np.float32)
        Y = np.empty((self.batch_size, n_classes_, imshape_[1], imshape_[2]),  dtype=np.float32)
        
        for i, (im_path, annot_path) in enumerate(zip(image_paths, annot_paths)):
            # Load image and annotation
            image = Image.open(im_path).convert("RGB")
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

            # Add to the batch
            X[i,] = image
            Y[i,] = mask

        return X, Y

    def create_multi_masks(self, anot_im):
        # Convert anot_im to numpy if it's a tensor
        if torch.is_tensor(anot_im):
            anot_im = anot_im.numpy()
            # Assuming anot_im is in CHW format, convert it to HWC for processing
            anot_im = anot_im.transpose(1, 2, 0)
        # Initialize channels
        background_mask = np.ones((anot_im.shape[0], anot_im.shape[1]), dtype=np.float32)
        channels = []
        for label, color in class_colors_.items():
            # Find pixels that math the color
            color = np.array(color, dtype=anot_im.dtype)
            tolerance = 1
            mask = np.all(np.abs(anot_im - color) <= tolerance, axis=-1)
            # Remove background pixels at the same location
            background_mask[mask] = 0
            # Add mask to channels
            mask = mask.astype(np.float32)
            channels.append(mask)

        # Add background mask to channels
        channels.append(background_mask)
        # Stack all channels to create multi-channel mask
        y = np.stack(channels, axis=0)
        return torch.tensor(y, dtype=torch.float32)

    def plot_multi_mask(masks):
        plt.figure(figsize=(n_classes_*2.5, 2))
        i = 1 # Iterable for subplot
        # To plot a combined color image later
        combined_color_image = np.zeros((*masks[0].shape, 3), dtype=np.uint8)
        for label, color in class_colors_.items():
            # Plot each mask
            plt.subplot(1, n_classes_ + 1, i)  # Adjust for an extra row for the combined color image
            plt.imshow(masks[i-1], cmap='gray')
            plt.title(label)
            
            # Update combined color image
            mask_indices = masks[i-1] > 0
            combined_color_image[mask_indices] = color

            i += 1
        
        # Adjust subplot for the background mask
        plt.subplot(1, n_classes_ + 1, i)
        plt.imshow(masks[-1], cmap='gray')
        plt.title('Background')
        
        # Adjust subplot for the combined color image
        plt.subplot(1, n_classes_ + 1, i+1)
        plt.imshow(combined_color_image)
        plt.title('Combined Color Image')
        
        # Increase vertical spacing and apply tight layout with padding
        plt.subplots_adjust(hspace=0.8, wspace=0.5)  # Adjusted hspace for increased spacing
        plt.tight_layout(pad=3.0)  # Apply tight layout with padding
        
        plt.show()

# Test code:
if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(image_paths=['Code/CNN/Sample_Data/polargeist/polargeist normal/2024 Jul 02 22-08-07241.png', 
                                         'Code/CNN/Sample_Data/polargeist/polargeist normal/2024 Jul 02 22-08-08242.png'],
                            annot_paths=['Code/CNN/Sample_Data/polargeist/polargeist hitbox/2024 Jul 02 21-44-574.png',
                                         'Code/CNN/Sample_Data/polargeist/polargeist hitbox/2024 Jul 02 21-44-585.png'],
                            transform=transform)
    a_img = Image.open('Code/CNN/Sample_Data/polargeist/polargeist hitbox/2024 Jul 02 21-44-574.png').convert("RGB")
    to_tensor_transform = transforms.ToTensor()
    a_img = to_tensor_transform(a_img) * 255.0
    f = dataset.create_multi_masks(a_img)
    CustomDataset.plot_multi_mask(f)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    label, item = next(iter(dataloader))
    CustomDataset.plot_multi_mask(item[0][0])