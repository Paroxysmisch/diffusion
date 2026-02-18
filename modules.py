import random

import diffusers
import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPTextModel, CLIPTokenizer

import padre


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

        temb_dim = 128 * 4

        for _, module in self.model.named_modules():
            # In diffusers, the class name is usually 'Attention'
            if module.__class__.__name__ == "Attention":
                padre.inject_multi_modal_padre(
                    module, degree=3, conv_kernel=3, temb_dim=temb_dim
                )

        self.model = torch.compile(self.model)
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
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=False)
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
        return (image * 255).to(torch.uint8)

    def on_train_start(self):
        # Prime FID with real images from the train dataloader once
        # This ensures the 'real' distribution is set before any validation runs
        print("Priming FID with real images...")
        train_loader = self.trainer.train_dataloader

        # Grab a few batches of real images
        for i, batch in enumerate(train_loader):
            if i >= 10:
                break  # 10 batches is plenty for a feature=64 proxy

            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            real_uint8 = ((images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)

            # Ensure images are on the correct device for the Inception model
            self.fid.update(real_uint8.to(self.device), real=True)

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
        if self.current_epoch > 0:
            self.fid.reset()
            fake_images = self.forward(batch_size=64)
            # Ensure fake images are moved to device
            self.fid.update(fake_images.to(self.device), real=False)

            fid_score = self.fid.compute()
            self.log("val/fid", fid_score, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # DDPM used Adam with a constant 2e-4 learning rate
        return torch.optim.Adam(self.parameters(), lr=2e-4)


class UNet2DConditionDiffusionModel(L.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()

        # 1. Load Text Encoder and Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.text_encoder.requires_grad_(False)  # Keep the encoder frozen

        # 2. Configure UNet2DConditionModel
        # This model uses cross-attention to "look" at text embeddings
        self.model = diffusers.UNet2DConditionModel(
            sample_size=hyperparameters.resolution // 8,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=512,  # Must match CLIP-base output dim
            dropout=0.1,
        )

        self.model = torch.compile(self.model)
        self.hyperparameters = hyperparameters

        # In __init__
        self.vae = diffusers.AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.requires_grad_(False)  # Keep the VAE frozen

        self.scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

        self.fid = FrechetInceptionDistance(feature=192, reset_real_features=False)
        self.save_hyperparameters()

    def _get_text_embeddings(self, prompts):
        # Helper to convert strings to CLIP embeddings
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Output shape: [batch, sequence_length, 512]
        return self.text_encoder(inputs.input_ids)[0]

    @torch.no_grad()
    def forward(self, prompts, guidance_scale=7.5):
        """Sampling Loop with Classifier-Free Guidance in Latent Space"""
        device = self.device
        batch_size = len(prompts)

        # 1. Prepare conditional and unconditional text embeddings
        # We concatenate them to run them through the UNet in a single batch for efficiency
        cond_embeddings = self._get_text_embeddings(prompts)
        uncond_embeddings = self._get_text_embeddings([""] * batch_size)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # 2. Setup Scheduler and Initial Latents
        self.scheduler.set_timesteps(1000)
        latent_size = self.model.config.sample_size
        # Usually 4 channels for VAE latents
        latents = torch.randn((batch_size, 4, latent_size, latent_size), device=device)

        # 3. Diffusion Iteration
        for t in self.scheduler.timesteps:
            # Expand latents for dual pass: [uncond_latents, cond_latents]
            # This allows us to predict noise for both prompts simultaneously
            latent_model_input = torch.cat([latents] * 2)

            # Predict noise residual
            model_output = self.model(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Separate the prediction into unconditional and conditional parts
            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)

            # Apply Classifier-Free Guidance formula:
            # noise = uncond + scale * (cond - uncond)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Update latents using the scheduler
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 4. Decode latents back to pixels using VAE
        # Apply the scaling factor (standard for Stable Diffusion-like models)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        # 5. Normalize and convert to [0, 255] uint8
        image = (image / 2 + 0.5).clamp(0, 1)
        return (image * 255).to(torch.uint8)

    def on_train_start(self):
        # Prime FID with real images from the train dataloader once
        # FID only needs the images to establish the 'real' distribution baseline
        print("Priming FID with real images...")

        # Access the dataloader through Lightning's trainer
        train_loader = self.trainer.train_dataloader

        for i, batch in enumerate(train_loader):
            if i >= 40:
                break

            # In your text-conditioned setup, batch is (images, prompts)
            # We only need the images (index 0)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch

            # Rescale from [-1, 1] to [0, 1] and convert to uint8 as required by torchmetrics FID
            real_uint8 = ((images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)

            # Update the FID metric with real samples
            # The metric handles the internal InceptionV3 forward pass
            self.fid.update(real_uint8.to(self.device), real=True)

    def training_step(self, batch, batch_idx):
        images, prompts = batch  # Assumes dataloader returns (Image, String)

        # Classifier-Free Guidance
        p_drop = 0.1
        prompts = [p if random.random() > p_drop else "" for p in prompts]

        with torch.no_grad():
            # latents shape is roughly [batch, 4, H/8, W/8]
            latents = self.vae.encode(images).latent_dist.sample()
            # Scale latents (Stable Diffusion uses a scaling factor of self.vae.config.scaling_factor)
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, 1000, (latents.shape[0],), device=self.device
        ).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get text conditioning
        encoder_hidden_states = self._get_text_embeddings(prompts)

        # Predict noise
        noise_pred = self.model(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(prompts))
        return loss

    def validation_step(self, batch, batch_idx):
        images, prompts = batch
        encoder_hidden_states = self._get_text_embeddings(prompts)

        with torch.no_grad():
            # latents shape is roughly [batch, 4, H/8, W/8]
            latents = self.vae.encode(images).latent_dist.sample()
            # Scale latents (Stable Diffusion uses a scaling factor of self.vae.config.scaling_factor)
            latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.full((latents.shape[0],), 500, device=self.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.model(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample
        self.log("val/loss", F.mse_loss(noise_pred, noise), batch_size=len(prompts))

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            self.fid.reset()
            # Generate a small sample for FID using fixed prompts
            test_prompts = ["a blue bird"] * 32 + ["a red bird"] * 32
            fake_images = self.forward(test_prompts)
            self.fid.update(fake_images.to(self.device), real=False)
            self.log(
                "val/fid",
                self.fid.compute(),
                prog_bar=True,
                sync_dist=True,
                batch_size=len(test_prompts),
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=4e-5)


class UNet2DConditionPixelDiffusionModel(L.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()

        # 1. Load Text Encoder and Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.text_encoder.requires_grad_(False)  # Keep the encoder frozen

        # 2. Configure UNet2DConditionModel
        # This model uses cross-attention to "look" at text embeddings
        self.model = diffusers.UNet2DConditionModel(
            sample_size=hyperparameters.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=512,  # Must match CLIP-base output dim
            dropout=0.1,
        )

        self.model = torch.compile(self.model)
        self.hyperparameters = hyperparameters

        self.scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
        )

        self.fid = FrechetInceptionDistance(feature=192, reset_real_features=False)
        self.save_hyperparameters()

    def _get_text_embeddings(self, prompts):
        # Helper to convert strings to CLIP embeddings
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Output shape: [batch, sequence_length, 512]
        return self.text_encoder(inputs.input_ids)[0]

    @torch.no_grad()
    def forward(self, prompts, guidance_scale=7.5):
        """Sampling Loop with Classifier-Free Guidance"""
        device = self.device
        batch_size = len(prompts)

        # 1. Prepare conditional embeddings
        cond_embeddings = self._get_text_embeddings(prompts)

        # 2. Prepare unconditional (null) embeddings
        # Using empty strings as the null prompt
        uncond_embeddings = self._get_text_embeddings([""] * batch_size)

        # Concatenate for a single batch pass: [Uncond, Cond]
        # This doubles the effective batch size for the UNet
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        self.scheduler.set_timesteps(1000)
        latent_size = self.model.config.sample_size
        images = torch.randn((batch_size, 3, latent_size, latent_size), device=device)

        for t in self.scheduler.timesteps:
            # Expand images for classifier-free guidance pass
            # [batch, 3, H, W] -> [batch * 2, 3, H, W]
            model_input = torch.cat([images] * 2)

            # Predict noise
            model_output = self.model(
                model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # Split output into unconditional and conditional
            noise_pred_uncond, noise_pred_cond = model_output.chunk(2)

            # Perform Guidance: noise = uncond + scale * (cond - uncond)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # Compute previous noisy sample
            images = self.scheduler.step(noise_pred, t, images).prev_sample

        # Post-process
        images = (images / 2 + 0.5).clamp(0, 1)
        return (images * 255).to(torch.uint8)

    def on_train_start(self):
        # Prime FID with real images from the train dataloader once
        # FID only needs the images to establish the 'real' distribution baseline
        print("Priming FID with real images...")

        # Access the dataloader through Lightning's trainer
        train_loader = self.trainer.train_dataloader

        for i, batch in enumerate(train_loader):
            if i >= 10:
                break

            # In your text-conditioned setup, batch is (images, prompts)
            # We only need the images (index 0)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch

            # Rescale from [-1, 1] to [0, 1] and convert to uint8 as required by torchmetrics FID
            real_uint8 = ((images / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)

            # Update the FID metric with real samples
            # The metric handles the internal InceptionV3 forward pass
            self.fid.update(real_uint8.to(self.device), real=True)

    def training_step(self, batch, batch_idx):
        images, prompts = batch  # Assumes dataloader returns (Image, String)

        # Classifier-Free Guidance
        p_drop = 0.1
        prompts = [p if random.random() > p_drop else "" for p in prompts]

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (images.shape[0],),
            device=self.device,
        ).long()
        noisy_latents = self.scheduler.add_noise(images, noise, timesteps)

        # Get text conditioning
        encoder_hidden_states = self._get_text_embeddings(prompts)

        # Predict noise
        noise_pred = self.model(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = F.mse_loss(noise_pred, noise)
        self.log("train/loss", loss, prog_bar=True, batch_size=len(prompts))
        return loss

    def validation_step(self, batch, batch_idx):
        images, prompts = batch
        encoder_hidden_states = self._get_text_embeddings(prompts)

        noise = torch.randn_like(images)
        timesteps = torch.full((images.shape[0],), 500, device=self.device).long()

        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        noise_pred = self.model(
            noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        self.log("val/loss", loss, batch_size=len(prompts))

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            self.fid.reset()
            # Generate a small sample for FID using fixed prompts
            test_prompts = ["a dog"] * 32 + ["an airplane"] * 32
            fake_images = self.forward(test_prompts)
            self.fid.update(fake_images.to(self.device), real=False)
            self.log(
                "val/fid",
                self.fid.compute(),
                prog_bar=True,
                sync_dist=True,
                batch_size=len(test_prompts),
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=5e-5)
