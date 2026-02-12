import argparse
import os

import torch
from torchvision.utils import save_image

from main import Hyperparameters
from modules import UNet2DConditionDiffusionModel


def run_inference(checkpoint_path, output_dir, batch_size, num_images, prompt):
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("medium")

    # 2. Load the model from checkpoint
    # We use map_location to ensure it loads correctly regardless of where it was saved
    model = UNet2DConditionDiffusionModel.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.to(device)
    model.eval()

    # 3. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Generate images
    print(f"Generating {num_images} images...")

    # We loop in case num_images > batch_size to avoid OOM
    count = 0
    with torch.no_grad():
        while count < num_images:
            current_batch = min(batch_size, num_images - count)

            # The forward method returns uint8 [0, 255]
            # torchvision.utils.save_image expects floats [0, 1]
            images_uint8 = model(batch_size=prompt * current_batch)
            images_float = images_uint8.float() / 255.0

            for i in range(images_float.size(0)):
                img_path = os.path.join(output_dir, f"sample_{count}.png")
                save_image(images_float[i], img_path)
                count += 1

            print(f"Saved {count}/{num_images} images")

    print(f"Inference complete. Images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Inference Script")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--out", type=str, default="./outputs", help="Directory to save images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generation"
    )
    parser.add_argument(
        "--num_images", type=int, default=64, help="Total number of images to generate"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="an airplane",
        help="Text prompt to condition image generation",
    )

    args = parser.parse_args()
    run_inference(args.ckpt, args.out, args.batch_size, args.num_images, args.prompt)
