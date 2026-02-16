from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from train_conditional_cub import DetailedTextConditionedCUB


def denormalize(tensor):
    """Reverses the (-1, 1) normalization for visualization."""
    return (tensor * 0.5) + 0.5


def test_sampling():
    # 1. Setup Paths (Update these to your actual paths)
    IMG_DIR = "./data/CUB_200_2011/images"
    TEXT_DIR = "./data/cvpr2016_cub/text_c10"
    RESOLUTION = 256

    transform = transforms.Compose(
        [
            transforms.Resize(RESOLUTION),
            transforms.CenterCrop(RESOLUTION),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # 2. Initialize Dataset
    dataset = DetailedTextConditionedCUB(IMG_DIR, TEXT_DIR, transform=transform)
    print(f"Dataset initialized with {len(dataset)} images.")

    # 3. Create DataLoader
    batch_size = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Grab one batch
    images, prompts = next(iter(dataloader))

    # 5. Visualize
    fig, axes = plt.subplots(1, batch_size, figsize=(20, 5))

    for i in range(batch_size):
        # Convert tensor to numpy and reorder dims from (C, H, W) to (H, W, C)
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  # Ensure range is exactly [0, 1]

        axes[i].imshow(img)
        # Wrap text so it doesn't run off the side
        wrapped_prompt = "\n".join(
            [prompts[i][j : j + 30] for j in range(0, len(prompts[i]), 30)]
        )
        axes[i].set_title(wrapped_prompt, fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("cub_sample_test.png")
    print("Test image saved as 'cub_sample_test.png'")
    plt.show()


if __name__ == "__main__":
    test_sampling()
