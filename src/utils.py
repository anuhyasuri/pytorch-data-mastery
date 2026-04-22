import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_data(dataloader, class_names, is_hf=False):
    """
    Pulls one batch of dataloader to visualize images and labels 
    from either Pytorch or Hugging Face
    """
    if is_hf == True:
        #Hugging Face dataloader returns a dictionary
        images = dataloader["pixel_values"]
        labels = dataloader["label"]
    else:
        #Kaggle dataloader returns a tuple
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
    if is_hf == True:
         plt.savefig('hf_data.png')
    else:
        plt.savefig('kaggle_data.png')