import dataclasses
import random

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from main import EMACallback
from modules import UNet2DConditionPixelDiffusionModel


@dataclasses.dataclass
class Hyperparameters:
    batch_size: int = 512
    num_timesteps: int = 1000
    resolution: int = 32


class TextConditionedCIFAR(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset):
        self.cifar = cifar_dataset
        # Standard CIFAR-10 class names
        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, label_idx = self.cifar[idx]
        # Transform the integer into a prompt
        templates = [
            "a photo of a {}",
            "a pixel art {}",
            "a small {}",
            "a {} in a square",
        ]
        prompt = random.choice(templates).format(self.classes[label_idx])
        return image, prompt


def main():
    torch.set_float32_matmul_precision("medium")
    hyperparameters = Hyperparameters()
    wandb_logger = WandbLogger(
        project="diffusion-conditional-cifar10", entity="paroxysmisch-university-of-cambridge"
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    base_train = datasets.CIFAR10(
        root="data/", train=True, download=True, transform=transform
    )
    base_val = datasets.CIFAR10(
        root="data/", train=False, download=True, transform=transform
    )

    train_dataset = TextConditionedCIFAR(base_train)
    validation_dataset = TextConditionedCIFAR(base_val)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="diffusion-conditional-cifar10-{epoch:04d}",
        every_n_epochs=10,
        monitor="val/loss",
        mode="min",
        save_top_k=10,  # Set to -1 to keep all checkpoints, or n to keep the n best
    )

    ema_callback = EMACallback(decay=0.9999)

    trainer = L.Trainer(
        max_epochs=250,
        logger=wandb_logger,
        check_val_every_n_epoch=10,
        accelerator="cuda",
        callbacks=[checkpoint_callback, ema_callback],
    )
    model = UNet2DConditionPixelDiffusionModel(hyperparameters)
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
