import matplotlib.pyplot as plt
import numpy as np


def visualize(train_loader):
    images, labels = next(iter(train_loader))

    # denormalize برای نمایش
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denormalize(img_tensor):
        img = img_tensor.numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    plt.figure(figsize=(10, 6))
    for i in range(8):
        img = denormalize(images[i])
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Labels: {labels[i].numpy().astype(int)}")
        plt.axis("off")

    plt.show()
