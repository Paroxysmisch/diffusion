import diffusers
import numpy as np
import torch
from PIL import Image

from main import Hyperparameters
from modules import UNet2DDiffusionModel

model = UNet2DDiffusionModel.load_from_checkpoint(
    # "./diffusion/4gsxqc4s/checkpoints/epoch=9-step=15630.ckpt"
    "./diffusion/7w5tmyq4/checkpoints/epoch=199-step=312600.ckpt"
)
hyperparameters = Hyperparameters()

device = torch.accelerator.current_accelerator()

model = model.model
model.eval()

scheduler = diffusers.DDPMScheduler(num_train_timesteps=hyperparameters.num_timesteps)
scheduler.set_timesteps(hyperparameters.num_timesteps)  # Number of inference steps

in_channels = 3
sample_size = 32
num_images = 32

for i in range(num_images):
    # Start from pure noise
    sample_shape = (
        1,
        in_channels,
        sample_size,
        sample_size,
    )
    image = torch.randn(sample_shape).to(model.device)

    for t in scheduler.timesteps:
        # Predict the noise residual
        with torch.no_grad():
            noise_pred = model(image, t).sample

        # Compute the previous image x_{t-1}
        image = scheduler.step(noise_pred, t, image).prev_sample

    # Convert to image (from [-1, 1] to [0, 255])
    image = (image.clamp(-1, 1) + 1) / 2  # [0,1]
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]  # CHW -> HWC
    image = (image * 255).astype(np.uint8)

    Image.fromarray(image).save(f"generated_{i}.png")
