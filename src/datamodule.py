import pytorch_lightning as pl
import numpy as np
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.dataset import CONICDataset

class CONICDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, biomarker, batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.biomarker = biomarker
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = self.get_train_transforms()
        self.val_transforms = self.get_val_transforms()
        
    def prepare_data(self):
        # Load data
        images = np.load(os.path.join(self.data_dir, 'images.npy'))
        labels = np.load(os.path.join(self.data_dir, 'labels.npy'))
        # Keep only data with the chosen biomarker
        is_biomarker = np.any(labels[..., 1]==self.biomarker, axis=(1, 2))
        biomarker_idxs = np.where(is_biomarker) # np.where() without condition just return the indexes of the non-zero elements
        images, labels = images[biomarker_idxs], labels[biomarker_idxs]
        # Keep only semantic segmentation channel and turn to binary masks
        labels = labels[..., 1]
        labels = np.where(labels==self.biomarker, 1, 0)
        self.images = images
        self.labels = labels
        # Hold-out idxs
        idxs = np.arange(start=0, stop=len(self.images))
        self.train_idxs, self.val_idxs = train_test_split(idxs, test_size=0.2, random_state=42)
        
    def setup(self, stage):
        if stage=='fit':
            train_images, train_labels = self.images[self.train_idxs], self.labels[self.train_idxs]
            self.train_ds = CONICDataset(images=train_images, masks=train_labels, transforms=self.train_transforms)
            val_images, val_labels = self.images[self.val_idxs], self.labels[self.val_idxs]
            self.val_ds = CONICDataset(images=val_images, masks=val_labels, transforms=self.val_transforms)
            
        if stage=='test':
            pass

    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        pass
        
    def get_train_transforms(self): 
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ])
        return transforms

    def get_val_transforms(self):
        transforms = A.Compose([
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ])
        return transforms
        
        