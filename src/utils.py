import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_data(dataloader, class_names):
    """
    Pulls one batch of dataloader to visualize images and labels
    """
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(8,8))

    for i in range(8):
        plt.subplot(2,4, i+1)
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(class_names[labels[i].item()])
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('kaggle_data.png')