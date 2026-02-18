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


import torch
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    def __init__(self, decay=0.9999, use_num_updates=True):
        super().__init__()
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.ema_params = {}
        self.num_updates = 0
        self.is_swapped = False

    def _init_ema(self, pl_module):
        # Initialize EMA weights on the same device as the model
        self.ema_params = {
            name: p.data.clone().detach().to(pl_module.device)
            for name, p in pl_module.named_parameters()
            if p.requires_grad
        }

    def on_fit_start(self, trainer, pl_module):
        if not self.ema_params:
            self._init_ema(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.num_updates += 1
        decay = self.decay
        if self.use_num_updates:
            # Dynamic decay: starts low to follow model, ends high to stabilize
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if name in self.ema_params:
                    ema_p = self.ema_params[name]
                    # In-place update: ema = ema * decay + p * (1 - decay)
                    ema_p.mul_(decay).add_(p.data, alpha=1 - decay)

    def swap_weights(self, pl_module):
        """Swaps model weights with EMA weights in-place to save memory."""
        for name, p in pl_module.named_parameters():
            if name in self.ema_params:
                tmp = p.data.clone()
                p.data.copy_(self.ema_params[name])
                self.ema_params[name].copy_(tmp)
        self.is_swapped = not self.is_swapped

    # Evaluation & Inference Hooks
    def on_validation_start(self, trainer, pl_module):
        if self.ema_params and not self.is_swapped:
            self.swap_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self.is_swapped:
            self.swap_weights(pl_module)

    def on_test_start(self, trainer, pl_module):
        if self.ema_params and not self.is_swapped:
            self.swap_weights(pl_module)

    def on_test_end(self, trainer, pl_module):
        if self.is_swapped:
            self.swap_weights(pl_module)

    def on_predict_start(self, trainer, pl_module):
        if self.ema_params and not self.is_swapped:
            self.swap_weights(pl_module)

    def on_predict_end(self, trainer, pl_module):
        if self.is_swapped:
            self.swap_weights(pl_module)

    # Persistence & Device Management
    def state_dict(self):
        return {
            "ema_params": self.ema_params,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict):
        self.ema_params = state_dict["ema_params"]
        self.num_updates = state_dict["num_updates"]

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        device = pl_module.device
        for k in self.ema_params:
            self.ema_params[k] = self.ema_params[k].to(device)


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
        max_epochs=750,
        logger=wandb_logger,
        check_val_every_n_epoch=10,
        accelerator="cuda",
        callbacks=[checkpoint_callback, ema_callback],
    )
    model = UNet2DDiffusionModel(hyperparameters)
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    main()
