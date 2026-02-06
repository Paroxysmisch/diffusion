import dataclasses

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision import datasets, transforms

from modules import UNet2DDiffusionModel


@dataclasses.dataclass
class Hyperparameters:
    batch_size: int = 256
    num_timesteps: int = 1000


class EMACallback(Callback):
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        self.original_weights = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema_weights is None:
            # Initialize on the same device as the model
            self.ema_weights = {
                k: v.clone().detach() for k, v in pl_module.state_dict().items()
            }
            return

        with torch.no_grad():
            for k, v in pl_module.state_dict().items():
                # In-place update is fast on GH200
                self.ema_weights[k].copy_(
                    self.decay * self.ema_weights[k] + (1 - self.decay) * v
                )

    def on_validation_start(self, trainer, pl_module):
        if self.ema_weights is not None:
            # Store current weights on CPU to save GPU VRAM
            self.original_weights = {
                k: v.cpu().clone() for k, v in pl_module.state_dict().items()
            }
            # Load EMA weights (ensure they stay on GPU if that's where pl_module is)
            pl_module.load_state_dict(self.ema_weights, strict=True)

    def on_validation_end(self, trainer, pl_module):
        # Only swap back if we actually performed a swap in on_validation_start
        if self.original_weights is not None:
            pl_module.load_state_dict(self.original_weights)
            self.original_weights = None  # Clear ref to free CPU memory

def main():
    torch.set_float32_matmul_precision("medium")
    hyperparameters = Hyperparameters()
    wandb_logger = WandbLogger(
        project="diffusion", entity="paroxysmisch-university-of-cambridge"
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        "data/",
        train=True,
        download=True,
        transform=transform,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
    )

    validation_dataset = datasets.CIFAR10(
        "data/",
        train=False,
        download=True,
        transform=transform,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=hyperparameters.batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="diffusion-cifar10-{epoch:04d}",
        every_n_epochs=10,
        monitor="val/loss",
        mode="min",
        save_top_k=5,  # Set to -1 to keep all checkpoints, or n to keep the n best
    )

    ema_callback = EMACallback(decay=0.9999)

    trainer = L.Trainer(
        max_epochs=2000,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        accelerator="cuda",
        callbacks=[checkpoint_callback, ema_callback],
    )
    model = UNet2DDiffusionModel(hyperparameters)
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
