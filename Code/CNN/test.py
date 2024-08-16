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

k = torch.tensor([[1,0,0],[0,1,0],[0,0,1]]).float()
m = F.softmax(k, dim=0)
print(m)
max_vals, _ = torch.max(m, dim=1, keepdim=True)
binary_mask = (m == max_vals).float()
print(max_vals)
print(binary_mask)

f = torch.tensor([])