import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from utils import visualize_results

def dice_coefficient(pred, target):
    smooth = 1e-5
    num_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=1)
    dice_scores = []

    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return np.mean(dice_scores)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, accumulation_steps=4):
    scaler = GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        optimizer.zero_grad()
        for i, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, masks = images.to(device), masks.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            train_dice += dice_coefficient(outputs, masks)
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks)
                
                if i == 0 and epoch % 5 == 0:
                    visualize_results(
                        images[0, 0].cpu().numpy(),
                        masks[0].cpu().numpy(),
                        outputs[0].argmax(dim=0).cpu().numpy()
                    )
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model_checkpoint.pth')