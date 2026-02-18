import dataclasses
import os
import random
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (Callback, ModelCheckpoint,
                                         WeightAveraging)
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from torch.optim.swa_utils import get_ema_avg_fn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import wandb
from modules import UNet2DConditionDiffusionModel


@dataclasses.dataclass
class Hyperparameters:
    batch_size: int = 128
    num_timesteps: int = 1000
    resolution: int = 256


class EMACallback(WeightAveraging):
    def __init__(self, decay=0.9999):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay))

    def should_update(self, step_idx=None, epoch_idx=None):
        # Start after 10 epochs.
        return (epoch_idx is not None) and (epoch_idx >= 10)


class DetailedTextConditionedCUB(torch.utils.data.Dataset):
    def __init__(self, img_root, text_root, transform=None):
        self.img_root = Path(img_root)
        self.text_root = Path(text_root)
        self.transform = transform

        # Gather all image paths
        self.image_paths = sorted(list(self.img_root.glob("**/*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # 1. Load Image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 2. Map Image path to Text path
        # Example: .../images/001.Black_footed_Albatross/Bird_0001.jpg
        # -> .../text_c10/001.Black_footed_Albatross/Bird_0001.txt
        rel_path = img_path.relative_to(self.img_root)
        text_path = self.text_root / rel_path.with_suffix(".txt")

        # 3. Randomly pick one description from the file
        try:
            with open(text_path, "r") as f:
                descriptions = [line.strip() for line in f.readlines() if line.strip()]
            prompt = random.choice(descriptions)
        except FileNotFoundError:
            # Fallback if text is missing: use cleaned folder name
            prompt = rel_path.parent.name.split(".")[-1].replace("_", " ").lower()

        return image, prompt


class DiffusionVisualizerCallback(Callback):
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        # Fixed prompts to track progress consistently
        self.test_prompts = [
            "a photo of a bright red bird with a black beak",
            "a small blue bird perched on a branch",
            "a large yellow bird with long wings",
            "a bird with brown feathers and a white breast",
        ][:num_samples]

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure we are in eval mode
        pl_module.eval()

        # Generate images using the CFG forward pass
        # This will use the guidance_scale=7.5 by default
        images_uint8 = pl_module(self.test_prompts)

        # Convert to a format Wandb likes (List of Wandb Images)
        images_to_log = [
            wandb.Image(img.permute(1, 2, 0).cpu().numpy(), caption=prompt)
            for img, prompt in zip(images_uint8, self.test_prompts)
        ]

        # Log to the existing wandb logger
        trainer.logger.experiment.log(
            {
                "samples/generated_birds": images_to_log,
                "global_step": trainer.global_step,
            }
        )

        pl_module.train()


def main():
    torch.set_float32_matmul_precision("medium")
    hp = Hyperparameters()

    # Define paths based on your environment
    IMG_DIR = "./data/CUB_200_2011/images"
    TEXT_DIR = "./data/cvpr2016_cub/text_c10"

    transform = transforms.Compose(
        [
            transforms.Resize(hp.resolution),
            transforms.CenterCrop(hp.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_dataset = DetailedTextConditionedCUB(IMG_DIR, TEXT_DIR, transform=transform)

    # Standard 90/10 split
    indices = list(range(len(full_dataset)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    split = int(0.9 * len(full_dataset))

    train_ds = Subset(full_dataset, indices[:split])
    val_ds = Subset(full_dataset, indices[split:])

    train_loader = DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp.batch_size, shuffle=False, num_workers=32, pin_memory=True
    )

    wandb_logger = WandbLogger(
        project="diffusion-conditional-cub",
        entity="paroxysmisch-university-of-cambridge",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="diffusion-conditional-cub-{epoch:04d}",
        every_n_epochs=10,
        monitor="val/loss",
        mode="min",
        save_top_k=10,  # Set to -1 to keep all checkpoints, or n to keep the n best
    )

    ema_callback = EMACallback(decay=0.9999)

    visualizer_callback = DiffusionVisualizerCallback(num_samples=4)

    trainer = L.Trainer(
        max_epochs=750,
        logger=wandb_logger,
        check_val_every_n_epoch=10,
        accelerator="cuda",
        callbacks=[checkpoint_callback, ema_callback, visualizer_callback],
    )

    model = UNet2DConditionDiffusionModel(hp)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
