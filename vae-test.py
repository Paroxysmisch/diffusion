import os

import torch
import torchvision
from torchvision import datasets, transforms

# Import your model class and Hyperparameters
from main import Hyperparameters
from modules import UNet2DConditionDiffusionModel
from train_conditional import coco_collate_fn


def test_vae_coco_reconstruction(
    checkpoint_path, coco_root, coco_ann_file, output_path="vae_coco_test.png"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = UNet2DConditionDiffusionModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.eval()
    vae = model.vae.to(device)
    scaling_factor = vae.config.scaling_factor

    # 2. Prepare COCO transforms
    # We use 256x256 as a standard for Stable Diffusion style testing
    transform = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Crucial [-1, 1] range
        ]
    )

    # 3. Load COCO Validation set (or train, doesn't matter for VAE test)
    try:
        dataset = datasets.CocoCaptions(
            root=coco_root, annFile=coco_ann_file, transform=transform
        )
    except Exception as e:
        print(f"Error loading COCO: {e}. Ensure paths are correct.")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=coco_collate_fn
    )

    # Get a batch
    # CocoCaptions returns (image, [list of captions])
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    # 4. VAE Roundtrip
    with torch.no_grad():
        # Encode -> Latent (Space is now 32x32)
        latents = vae.encode(real_images).latent_dist.sample()

        # Scaling logic
        scaled_latents = latents * scaling_factor
        unscaled_latents = (1 / scaling_factor) * scaled_latents

        # Decode -> Image
        reconstruction = vae.decode(unscaled_latents).sample

    # 5. Save comparison
    def denormalize(tensor):
        return (tensor / 2 + 0.5).clamp(0, 1)

    comparison = torch.cat(
        [denormalize(real_images), denormalize(reconstruction)], dim=0
    )
    grid = torchvision.utils.make_grid(comparison, nrow=4)

    torchvision.utils.save_image(grid, output_path)
    print(f"COCO VAE test complete. Comparison saved to {output_path}")


if __name__ == "__main__":
    CKPT = "./checkpoints/diffusion-conditional-cifar10-epoch=0589.ckpt"
    COCO_IMG = "./data/val2017"
    COCO_ANN = "./data/annotations/captions_val2017.json"

    test_vae_coco_reconstruction(CKPT, COCO_IMG, COCO_ANN)
