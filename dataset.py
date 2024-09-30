import os
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage import rotate

class PENGWINDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=64, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.augment = augment
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.mha')])

    def __len__(self):
        return len(self.image_files)

    def augment_data(self, image, mask):
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-20, 20)
            image = rotate(image, angle, axes=(1, 2), reshape=False, mode='nearest')
            mask = rotate(mask, angle, axes=(1, 2), reshape=False, mode='nearest')

        # Random flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        return image, mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])

        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        image = normalize_image(image)

        if self.augment:
            image, mask = self.augment_data(image, mask)

        # Extract random patch
        d, h, w = image.shape
        d_start = np.random.randint(0, d - self.patch_size + 1)
        h_start = np.random.randint(0, h - self.patch_size + 1)
        w_start = np.random.randint(0, w - self.patch_size + 1)

        image_patch = image[d_start:d_start+self.patch_size, 
                            h_start:h_start+self.patch_size, 
                            w_start:w_start+self.patch_size]
        mask_patch = mask[d_start:d_start+self.patch_size, 
                          h_start:h_start+self.patch_size, 
                          w_start:w_start+self.patch_size]

        image_patch = torch.from_numpy(image_patch).float().unsqueeze(0)
        mask_patch = torch.from_numpy(mask_patch).long()

        return image_patch, mask_patch

def normalize_image(image):
    min_val = np.percentile(image, 1)
    max_val = np.percentile(image, 99)
    image = np.clip(image, min_val, max_val)
    image = (image - min_val) / (max_val - min_val + 1e-8)
    return image