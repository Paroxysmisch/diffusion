import dataclasses
import random

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from main import EMACallback
from modules import UNet2DConditionDiffusionModel


@dataclasses.dataclass
class Hyperparameters:
    batch_size: int = 256
    num_timesteps: int = 1000
    resolution: int = 128


def coco_collate_fn(batch):
    """
    COCO returns a list of 5 captions per image.
    This function picks ONE random caption per image to train on.
    """
    images = []
    prompts = []
    for img, captions in batch:
        images.append(img)
        # Select one random caption from the 5 available
        prompts.append(random.choice(captions))

    return torch.stack(images), prompts


def main():
    torch.set_float32_matmul_precision("medium")
    hyperparameters = Hyperparameters()
    wandb_logger = WandbLogger(
        project="diffusion-coco", entity="paroxysmisch-university-of-cambridge"
    )

    # MS-COCO images are diverse sizes; we must resize and center crop
    transform = transforms.Compose(
        [
            transforms.Resize(hyperparameters.resolution),
            transforms.CenterCrop(hyperparameters.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CocoCaptions(
        root="data/train2017",
        annFile="data/annotations/captions_train2017.json",
        transform=transform,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=coco_collate_fn,  # Crucial for handling text lists
    )

    validation_dataset = datasets.CocoCaptions(
        root="data/val2017",
        annFile="data/annotations/captions_val2017.json",
        transform=transform,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=32,
        collate_fn=coco_collate_fn,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="diffusion-coco-{epoch:04d}",
        every_n_epochs=10,
        monitor="val/loss",
        mode="min",
        save_top_k=5,  # Set to -1 to keep all checkpoints, or n to keep the n best
    )

    ema_callback = EMACallback(decay=0.9999)

    trainer = L.Trainer(
        max_epochs=750,
        logger=wandb_logger,
        check_val_every_n_epoch=10,
        accelerator="cuda",
        callbacks=[checkpoint_callback, ema_callback],
    )
    model = UNet2DConditionDiffusionModel(hyperparameters)
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
