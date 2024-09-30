import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import UNet3D
from dataset import PENGWINDataset
from loss import CombinedLoss
from train import train

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 1
    learning_rate = 1e-4
    num_epochs = 50
    patch_size = 64
    accumulation_steps = 4

    # Data paths
    train_image_dir = "/path/to/PENGWIN_images"
    train_mask_dir = "/path/to/PENGWIN_labels"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets and dataloaders
    train_dataset = PENGWINDataset(train_image_dir, train_mask_dir, patch_size=patch_size, augment=True)
    val_dataset = PENGWINDataset(train_image_dir, train_mask_dir, patch_size=patch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=31).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)