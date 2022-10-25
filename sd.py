#!/usr/env/bin python

from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
from torchvision import transforms as tfms
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from torch import autocast
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class StableDiffusionPipe:
    def __init__(
        self, 
        tokenizer:str="openai/clip-vit-large-patch14",
        text_encoder:str="openai/clip-vit-large-patch14",
        vae="stabilityai/sd-vae-ft-ema",
        unet="CompVis/stable-diffusion-v1-4", scheduler='lms', beta_start=0.00085, beta_end=0.012
        ):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer, torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder, torch_dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained(vae, torch_dtype=torch.float16)
        self.unet = UNet2DConditionModel.from_pretrained(unet, subfolder="unet", torch_dtype=torch.float16)
        self.beta_start,self.beta_end = beta_start, beta_end

        if scheduler == 'lms':
            self.scheduler = LMSDiscreteScheduler(beta_start=self.beta_start, beta_end=self.beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)
        else:
            print("scheduler not supported")
    
    def to_device(self, device):
        self.text_encoder.to(device)
        self.vae.to(device)
        self.unet.to(device)


def load_models(device):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
    # vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(device)
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)
    beta_start,beta_end = 0.00085,0.012
    scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)
    return tokenizer, text_encoder, vae, unet, scheduler

def text_embedder(prompts, tokenizer, text_encoder, maxlen=None, device='cuda'):
    if maxlen is None: maxlen = tokenizer.model_max_length
    text_input = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0].half()
    return text_embeddings

def mk_imgs(images):
    res = []
    for image in images:
        image = (image/2+0.5).clamp(0,1).detach().cpu().permute(1, 2, 0).numpy()
        res.append(Image.fromarray((image*255).round().astype("uint8")))
    return res

def pil2latent(input_im, device='cuda'):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(device).half()*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents2pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@torch.no_grad()
def diffuse(prompts, pipe, latents=None, height=512, width=512, guidance=7.5, seed=100, steps=70, device='cuda'):

    # params
    bs = len(prompts)
    if seed: torch.manual_seed(seed)
    
    # text cond embedding
    text_embedding = text_embedder(prompts, pipe.tokenizer, pipe.text_encoder)
    uncond_embedding = text_embedder([""] * bs, pipe.tokenizer, pipe.text_encoder, maxlen=text_embedding.shape[1])
    emb = torch.cat([uncond_embedding, text_embedding])

    # scheduler
    pipe.scheduler.set_timesteps(steps)

    # prep cond latents
    if latents == None:
        latents = torch.randn((bs, pipe.unet.in_channels, height//8, width//8))

    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        latents = latents.to(device).half() * pipe.scheduler.init_noise_sigma

    # loop
    with autocast(device):
        for i,ts in enumerate(tqdm(pipe.scheduler.timesteps)):
            inp = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), ts)
            # predict noise residual
            with torch.no_grad(): 
                noise_pred_uncond,noise_pred_text = pipe.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
            # guidance
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
            # previous noisy sample
            latents = pipe.scheduler.step(noise_pred, ts, latents).prev_sample

    with torch.no_grad(): return pipe.vae.decode(1 / 0.18215 * latents).sample

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def animate(prompts, pipe, rootdir='.', name='animation', device='cuda', max_frames=10000, num_steps=5, quality=90, height=512, width=512):
    
    outdir = os.path.join(rootdir, name)
    os.makedirs(outdir, exist_ok=True)
    # iterate the loop
    frame_index = 0

    # sample source
    init1 = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)
    while frame_index < max_frames:

        # sample the destination
        init2 = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)

        for i, t in enumerate(np.linspace(0, 1, num_steps)):
            init = slerp(float(t), init1, init2)

            print("dreaming... ", frame_index)
            with autocast("cuda"):
                image = diffuse(prompts, pipe, latents=init)
            im = mk_imgs(image)
            outpath = os.path.join(outdir, 'frame%06d.jpg' % frame_index)
            im[0].save(outpath, quality=quality)
            frame_index += 1
        
        init1 = init2


if __name__ == "__main__":
    pass
    # prompts = ["a photograph of a mouse skiing", "a donkey flying an airplane"]
    # height = 512
    # width = 512
    # num_inference_steps = 70
    # guidance_scale = 7.5
    # batch_size = 1
    # bs = batch_size
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    # text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
    # vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16).to(device)
    # unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)
    # beta_start,beta_end = 0.00085,0.012
    # scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)

    # img = Image.open('obama.jpg').resize((512, 512))
    # latents = pil2latent(img)
    # image = latents2pil(latents)

    # print('start loop')
    # images = diffusion_loop(prompts, tokenizer, text_encoder, vae, unet, scheduler, g=7.5, seed=100, steps=70)
    # print('end loop')
    # imgs = mk_imgs(images)
    # imgs[0].save("test.jpg")

