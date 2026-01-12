import os
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pydicom
import cv2

from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# ==========================
# Device
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ==========================
# Data Processor
# ==========================
class RSNADataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'stage_2_train')
        self.test_dir = os.path.join(data_dir, 'stage_2_test')
        self.train_csv = os.path.join(data_dir, 'stage_2_train.csv')
        self.sample_csv = os.path.join(data_dir, 'stage_2_sample_submission.csv')

        # Debug info
        print(f"\nUsing data directory: {self.data_dir}")
        if os.path.exists(self.data_dir):
            print("Directory contents (first 10 items):")
            for item in os.listdir(self.data_dir)[:10]:
                item_path = os.path.join(self.data_dir, item)
                if os.path.isdir(item_path):
                    print(f"  - {item}/ (directory)")
                else:
                    print(f"  - {item}")
        else:
            print("  Directory not found!")

        # Verify key files
        print("\nChecking for key files:")
        print(f"  stage_2_train.csv: {'✓' if os.path.exists(self.train_csv) else '✗'}")
        print(f"  stage_2_train/: {'✓' if os.path.exists(self.train_dir) else '✗'}")
        print(f"  stage_2_test/: {'✓' if os.path.exists(self.test_dir) else '✗'}")

        # Image preprocessing parameters
        self.img_size = 224  # Standard ResNet input size
        self.window_center = 40
        self.window_width = 80

    def load_and_preprocess_labels(self):
        """Load and preprocess training labels, pivot to one row per image."""
        print("\nLoading training labels...")
        df = pd.read_csv(self.train_csv)
        print(f"Original dataframe shape: {df.shape}")

        # Extract image_id and hemorrhage_type
        df['image_id'] = df['ID'].apply(lambda x: x.split('_')[1])
        df['hemorrhage_type'] = df['ID'].apply(lambda x: x.split('_')[2])

        print(f"Unique image IDs: {df['image_id'].nunique()}")
        print(f"Unique hemorrhage types: {df['hemorrhage_type'].unique()}")

        # Remove duplicate combinations
        duplicates = df.groupby(['image_id', 'hemorrhage_type']).size()
        if (duplicates > 1).any():
            print(f"Found {(duplicates > 1).sum()} duplicate combinations, removing duplicates...")
            df = df.drop_duplicates(subset=['image_id', 'hemorrhage_type'], keep='first')
            print(f"After removing duplicates: {df.shape}")

        # Pivot to wide format: one row per image, columns = hemorrhage types
        df_pivot = df.pivot(index='image_id', columns='hemorrhage_type', values='Label')
        df_pivot = df_pivot.reset_index()

        print(f"Pivoted dataframe shape: {df_pivot.shape}")
        print(f"Columns: {df_pivot.columns.tolist()}")

        # Fill missing hemorrhage type columns with 0 if needed
        hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular',
                            'subarachnoid', 'subdural', 'any']
        for h_type in hemorrhage_types:
            if h_type in df_pivot.columns:
                df_pivot[h_type] = df_pivot[h_type].fillna(0)
            else:
                # اگر ستونی نبود، اضافه کن و صفر پر کن
                df_pivot[h_type] = 0

        # Class distribution
        print("\nClass distribution:")
        for h_type in hemorrhage_types:
            positive_cases = df_pivot[h_type].sum()
            print(f"  {h_type}: {positive_cases} positive "
                  f"({positive_cases / len(df_pivot) * 100:.2f}%)")

        return df_pivot

    def apply_windowing_optimized(self, img, center, width):
        """CT windowing."""
        img_min = center - width // 2
        img_max = center + width // 2
        return np.clip(img, img_min, img_max)

    def dicom_to_array_optimized(self, dicom_path, img_size=224):
        """Read DICOM, apply windowing & normalization, return 224x224x3 uint8."""
        try:
            dicom = pydicom.dcmread(dicom_path)
            img = dicom.pixel_array.astype(np.float32)

            # Apply rescale if present
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                img = img * dicom.RescaleSlope + dicom.RescaleIntercept

            # Windowing
            img = self.apply_windowing_optimized(img, self.window_center, self.window_width)

            # Normalize to 0–255
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

            # Resize
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

            # Make 3-channel
            img = np.stack([img, img, img], axis=-1)

            return img
        except Exception as e:
            # می‌تونی اینجا لاگ هم چاپ کنی اگر لازم شد
            # print(f"Error reading {dicom_path}: {e}")
            return None


# ==========================
# Dataset: لود DICOM روی پرواز
# ==========================
class RSNADicomDataset(Dataset):
    """
    Dataset که فقط DataFrame و مسیر DICOM را نگه می‌دارد
    و هر بار در __getitem__ تصویر را از دیسک می‌خواند.
    """

    def __init__(self, df, img_dir, hemorrhage_types, processor, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.hemorrhage_types = hemorrhage_types
        self.processor = processor
        self.transform = transform
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']

        dicom_path = os.path.join(self.img_dir, f'ID_{image_id}.dcm')

        img_array = self.processor.dicom_to_array_optimized(
            dicom_path, self.processor.img_size
        )

        if img_array is None:
            # اگر خراب بود، تصویر سیاه برگردون
            img_array = np.zeros((self.processor.img_size,
                                  self.processor.img_size,
                                  3), dtype=np.uint8)

        image = Image.fromarray(img_array)

        if self.transform:
            image = self.transform(image)

        # multi-label vector به ترتیب hemorrhage_types
        labels = row[self.hemorrhage_types].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        

        return image, labels


# ==========================
# Model: ResNet50 with frozen backbone
# ==========================
class FreezeResNet(nn.Module):
    def __init__(self, num_classes=6, freeze_backbone=True):
        super(FreezeResNet, self).__init__()
        self.backbone = models.resnet50(pretrained=True)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ==========================
# Transforms
# ==========================
def create_data_transforms():
    """Augmentations برای train و transform ساده برای val/test."""
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_transforms


# ==========================
# Find data directory (مثل کد خودت)
# ==========================
def find_data_directory():
    """
    این تابع دایرکتوری دیتا را پیدا می‌کند.
    می‌تونی مسیرهای بیشتری اضافه کنی.
    """
    possible_paths = [
        r"E:\Test",        # مسیر خودت
        r"/kaggle/input",  # اگر روی Kaggle باشی
    ]

    for path in possible_paths:
        if os.path.exists(path):
            train_csv = os.path.join(path, 'stage_2_train.csv')
            if os.path.exists(train_csv):
                print(f"Found data directory: {path}")
                return path

            # check subdirs
            for subdir in os.listdir(path):
                subpath = os.path.join(path, subdir)
                if os.path.isdir(subpath):
                    train_csv = os.path.join(subpath, 'stage_2_train.csv')
                    if os.path.exists(train_csv):
                        print(f"Found data directory: {subpath}")
                        return subpath

    print("Could not find data directory with stage_2_train.csv")
    return None


# ==========================
# Build Dataloaders (بدون لود کل تصاویر در RAM)
# ==========================
def build_dataloaders(use_sample=True, sample_size=100, batch_size=16):
    # پیدا کردن دیتا
    data_dir = find_data_directory()
    if data_dir is None:
        print("Please check your data directory structure.")
        return None, None, None, None

    processor = RSNADataProcessor(data_dir)
    df_labels = processor.load_and_preprocess_labels()

    hemorrhage_types = ['epidural', 'intraparenchymal', 'intraventricular',
                        'subarachnoid', 'subdural', 'any']

    # مسیر هر تصویر را اضافه کن
    df_labels['path'] = df_labels['image_id'].apply(
        lambda x: os.path.join(processor.train_dir, f'ID_{x}.dcm')
    )

    # فقط ردیف‌هایی که فایلشون واقعاً هست
    df_labels = df_labels[df_labels['path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"\nNumber of images with existing DICOM files: {len(df_labels)}")

    # نمونه‌گیری (برای کاهش حجم)
    if use_sample:
        df_labels = df_labels.sample(
            n=min(sample_size, len(df_labels)), random_state=42
        ).reset_index(drop=True)
        print(f"Using sample of {len(df_labels)} images")

    # Stratify روی ستون 'any'
    y_any = df_labels['any'].values

    train_df, val_df = train_test_split(
        df_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_any
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    train_transforms, val_transforms = create_data_transforms()

    train_dataset = RSNADicomDataset(
        train_df,
        img_dir=processor.train_dir,
        hemorrhage_types=hemorrhage_types,
        processor=processor,
        transform=train_transforms
    )

    val_dataset = RSNADicomDataset(
        val_df,
        img_dir=processor.train_dir,
        hemorrhage_types=hemorrhage_types,
        processor=processor,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # اگر ارور یا مصرف RAM زیاد دیدی، 0 کن
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if device.type == 'cuda' and 2 > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if device.type == 'cuda' and 2 > 0 else False
    )

    print("\nData loaders created:")
    print(f"  Train loader batches: {len(train_loader)}")
    print(f"  Val loader batches:   {len(val_loader)}")

    return train_loader, val_loader, hemorrhage_types, processor


# ==========================
# Simple training loop (اختیاری)
# ==========================
def train_model(num_epochs=3,
                use_sample=True,
                sample_size=100,
                batch_size=16,
                lr=1e-3):
    train_loader, val_loader, hemorrhage_types, processor = build_dataloaders(
        use_sample=use_sample,
        sample_size=sample_size,
        batch_size=batch_size
    )

    if train_loader is None:
        print("Dataloaders not created. Exiting.")
        return None, None, None

    num_classes = len(hemorrhage_types)
    model = FreezeResNet(num_classes=num_classes, freeze_backbone=True)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    return model, train_loader, val_loader


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    # اینجا می‌تونی پارامترها را عوض کنی
    model, train_loader, val_loader = train_model(
        num_epochs=3,
        use_sample=True,   # اگر False کنی، کل دیتا استفاده می‌شه (ممکنه خیلی سنگین بشه)
        sample_size=500,   # تعداد نمونه برای تست
        batch_size=16,
        lr=1e-3
    )
