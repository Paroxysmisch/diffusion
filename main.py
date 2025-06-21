import dataclasses

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torchvision import datasets, transforms

from modules import UNet2DDiffusionModel


@dataclasses.dataclass
class Hyperparameters:
    batch_size: int = 32
    num_timesteps: int = 250


def main():
    hyperparameters = Hyperparameters()
    wandb_logger = WandbLogger(
        project="diffusion", entity="paroxysmisch-university-of-cambridge"
    )

    train_dataset = datasets.CIFAR10(
        "data/",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=True,
    )

    validation_dataset = datasets.CIFAR10(
        "data/",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
    )

    trainer = L.Trainer(max_epochs=200, logger=wandb_logger, check_val_every_n_epoch=1)
    model = UNet2DDiffusionModel(hyperparameters)
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
