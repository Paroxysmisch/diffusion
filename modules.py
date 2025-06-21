import diffusers
import lightning as L
import torch


class UNet2DDiffusionModel(L.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        self.model = diffusers.UNet2DModel(
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            block_out_channels=(64, 128, 256),
        )
        self.hyperparameters = hyperparameters
        self.scheduler = diffusers.DDPMScheduler(hyperparameters.num_timesteps)
        self.save_hyperparameters()

    def training_step(self, images_batch, batch_idx):
        images_batch = images_batch[0]  # discard the labels
        noise = torch.randn_like(images_batch)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (images_batch.size(0),),
            device=images_batch.device,
        )
        noisy_images = self.scheduler.add_noise(images_batch, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)

        self.log("training/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, images_batch, batch_idx):
        images_batch = images_batch[0]  # discard the labels
        noise = torch.randn_like(images_batch)
        steps = torch.full(
            (images_batch.size(0),),
            self.scheduler.config.num_train_timesteps - 1,
            device=images_batch.device,
        )
        noisy_images = self.scheduler.add_noise(images_batch, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = torch.nn.functional.mse_loss(residual, noise)

        self.log("validation/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]
