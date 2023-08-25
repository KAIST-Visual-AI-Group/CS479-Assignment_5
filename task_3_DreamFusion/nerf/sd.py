from diffusers import (AutoencoderKL, DDIMScheduler, PNDMScheduler,
                       UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer, logging

# suppress partial model loading warning
logging.set_verbosity_error()

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version="2.1", hf_key=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f"[INFO] loading stable diffusion...")

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(
            self.device
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            model_key, subfolder="unet"
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f"[INFO] loaded stable diffusion!")

    def get_text_embeds(self, prompt, negative_prompt) -> Float[torch.Tensor, "2 D"]:
        # TODO: return text embeddings.
        # It should be a concatenation of an unconditional prompt's embedding and conditional prompt's embedding in order.

        uncond_embeddings = torch.rand(1, 768).to(self.device)
        text_embeddings = torch.randn(1, 768).to(self.device)
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(
        self,
        text_embeddings: Float[torch.Tensor, "2B D"],
        pred_rgb: Float[torch.Tensor, "B C H W"],
        guidance_scale=100,
    ):
        """
        Input:
            text_embeddings: concatenation of uncond and cond text embeddings for classifier-free guidance.
            pred_rgb: rendered_img with a shape of [B,3,512,512] to be updated.
            guidance_scale: classifier-free guidance weight.
        Output:
            loss_sds: loss for SDS update.            
        """
        # TODO: compute gradient of SDS and return the loss.
        loss_sds = 0.0
        return loss_sds

    def produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):

        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast("cuda"):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )["sample"]

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        return latents

    def decode_latents(self, latents: Float[torch.Tensor, "B 4 64 64"]):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs: Float[torch.Tensor, "B 3 512 512"]):
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents
