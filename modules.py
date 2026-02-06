import diffusers
import lightning as L
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


class UNet2DDiffusionModel(L.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        # DDPM paper configuration for CIFAR-10
        self.model = diffusers.UNet2DModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 256),  # [1, 2, 2, 2] multiplier
            down_block_types=(
                "DownBlock2D",  # 32x32: No attention
                "AttnDownBlock2D",  # 16x16: Attention used here
                "DownBlock2D",  # 8x8: No attention
                "DownBlock2D",  # 4x4: No attention
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",  # 16x16: Attention used here
                "UpBlock2D",
            ),
            dropout=0.1,  # DDPM uses 0.1 dropout
        )
        self.hyperparameters = hyperparameters

        # DDPM paper noise schedule: Linear 1e-4 to 0.02
        self.scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",  # Predicting the noise (L_simple)
        )
        # Initialize FID metric (32x32 for CIFAR, features=64 or 2048)
        # 2048 is standard, but 64 is faster for "proxy" checks
        self.fid = FrechetInceptionDistance(feature=64)
        self.save_hyperparameters()

    @torch.no_grad()
    def forward(self, batch_size):
        """Standard DDPM Sampling Loop (T -> 0)"""
        device = self.device
        self.scheduler.set_timesteps(1000)
        # Start from pure noise
        image = torch.randn((batch_size, 3, 32, 32), device=device)

        for t in self.scheduler.timesteps:
            # 1. Predict noise residual
            model_output = self.model(image, t).sample

            # 2. Compute less noisy image (reverse step)
            image = self.scheduler.step(model_output, t, image).prev_sample

        # Rescale from [-1, 1] to [0, 1] and then to uint8 for FID
        image = (image / 2 + 0.5).clamp(0, 1)
        return (image * 255).to(torch.uint8).cpu()

    def training_step(self, images_batch, batch_idx):
        # images_batch is often a list/tuple: [images, labels]
        images = (
            images_batch[0] if isinstance(images_batch, (list, tuple)) else images_batch
        )

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=self.device,
        ).long()

        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.model(noisy_images, timesteps).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("train/loss", loss, prog_bar=True)

        # Update FID with REAL images (only on the first few batches)
        if batch_idx < 10:
            real_uint8 = ((images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            self.fid.update(real_uint8, real=True)
        return loss

    def validation_step(self, images_batch, batch_idx):
        images = (
            images_batch[0] if isinstance(images_batch, (list, tuple)) else images_batch
        )
        noise = torch.randn_like(images)
        # Check performance at a specific high-noise timestep
        timesteps = torch.full((images.shape[0],), 500, device=self.device).long()

        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("val/loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        # 3. Compute FID ONCE at the end of the epoch
        if (self.current_epoch) % 10 == 0:
            fake_images = self.forward(batch_size=64)  # Generate 64 images
            self.fid.update(fake_images, real=False)

            fid_score = self.fid.compute()
            self.log("val/fid", fid_score, prog_bar=True)
            self.fid.reset()

    def configure_optimizers(self):
        # DDPM used Adam with a constant 2e-4 learning rate
        return torch.optim.Adam(self.parameters(), lr=2e-4)
