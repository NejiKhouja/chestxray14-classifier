
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# The 14 pathology classes, in a fixed canonical order. "No Finding" is excluded.
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
]
NUM_CLASSES = len(LABELS)

# ImageNet normalization (DenseNet was pretrained on these statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_image_index(image_root):
    """
    Walk image_root once and map every '*.png' filename -> absolute path.

    Works for any layout (Kaggle's images_XXX/images/, a flat folder, etc.)
    so we never have to guess the directory structure.
    """
    index = {}
    for dirpath, _, filenames in os.walk(image_root):
        for fn in filenames:
            if fn.lower().endswith('.png'):
                index[fn] = os.path.join(dirpath, fn)
    if not index:
        raise FileNotFoundError(f"No .png images found under {image_root}")
    return index


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None, image_index=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        # build (or reuse) the filename -> path lookup once, not per __getitem__
        self.image_index = image_index if image_index is not None else build_image_index(image_root)
        # cache label columns as a numpy matrix for fast row access
        self._labels = self.df[LABELS].values.astype('float32')
        self._ages = self.df['Patient Age'].values.astype('float32')
        self._genders = self.df['Patient Gender'].values.astype('float32')
        self._files = self.df['Image Index'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        path = self.image_index[self._files[idx]]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        target = torch.from_numpy(self._labels[idx])
        age    = torch.tensor([self._ages[idx]], dtype=torch.float32)
        gender = torch.tensor([self._genders[idx]], dtype=torch.float32)
        return image, age, gender, target


def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
