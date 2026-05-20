import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# the 15 conditions in a fixed order
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration',
    'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

IMAGE_DIRS = [
    f'images_{str(i).zfill(3)}/images' for i in range(1, 13)
]

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image(self, filename):
        for folder in IMAGE_DIRS:
            path = os.path.join(self.image_root, folder, filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Image not found: {filename}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load image
        img_path = self._find_image(row['Image Index'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # multi-label target vector
        label_str = row['Finding Labels']
        target = torch.zeros(len(LABELS), dtype=torch.float32)
        for i, label in enumerate(LABELS):
            if label in label_str:
                target[i] = 1.0

        # clinical features
        age = torch.tensor([row['Patient Age']], dtype=torch.float32)
        gender = torch.tensor([row['Patient Gender']], dtype=torch.float32)

        return image, age, gender, target


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # affine combines rotation + small translation + scale in one pass
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            # randomly erase a small patch — forces model to not rely on any single region
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.08)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
if __name__ == '__main__':
    dataset = ChestXrayDataset(
        csv_path=r'C:\Users\deadx\OneDrive\Desktop\ChestX-ray14\data\row\train.csv',
        image_root=r'C:\Users\deadx\OneDrive\Desktop\ChestX-ray14\data\row\archive',
        transform=get_transforms(train=True)
    )
    image, age, gender, target = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Age: {age}, Gender: {gender}")
    print(f"Target: {target}")
    print(f"Dataset size: {len(dataset)}")