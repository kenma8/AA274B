import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

IMG_SIZE = 299
LABELS = ["cat", "dog", "neg"]

class ImageDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.image_paths = []
        self.image_labels = []
        for label_idx, label in enumerate(labels):
            label_dir = os.path.join(img_dir, label)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.image_labels.append(label_idx)
        self.samples = len(self.image_paths)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.image_labels[idx]

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(np.array(image), dtype=torch.float32)  # Convert to tensor and permute
        label = torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        return image, label, img_path

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def transform_img(img_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(img_path).convert("RGB")    
    return transform(image)

def decode_jpeg(file_path):
     """Loads a JPEG image and returns a numpy array."""
     try:
         image = Image.open(file_path).convert("RGB")
     except FileNotFoundError:
         print(f"File not found: {file_path}")
         return None  # Or handle the error appropriately
     return np.array(image)

def normalize_image(image):
    image = (image / 127.5) - 1
    return image

def resize_image(image, img_size):
   image = transforms.Resize((img_size, img_size))(Image.fromarray(np.uint8(image)))
   return np.array(image)

def normalize_resize_image(image, img_size):
    return resize_image(normalize_image(image), img_size)