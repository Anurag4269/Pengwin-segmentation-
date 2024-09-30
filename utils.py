import matplotlib.pyplot as plt

def visualize_results(image, true_mask, pred_mask, slice_num=None):
    if slice_num is None:
        slice_num = image.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image[slice_num, :, :], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask[slice_num, :, :], cmap='nipy_spectral', interpolation='nearest')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask[slice_num, :, :], cmap='nipy_spectral', interpolation='nearest')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()