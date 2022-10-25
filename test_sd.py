from diffusers import StableDiffusionPipeline
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
from torchvision import transforms as tfms
from PIL import Image
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import random
from sd import mk_imgs, StableDiffusionPipe, animate, diffuse

# params
prompts = ["ultrarealistic teddy bear DJ mixing vinyl. dramatic lighting. #unrealengine"]
prompts = ["ultrarealistic ai robot portrait, #unrealengine, 8k"]
prompts = ["robot running towards its freedom, 8 k"]
prompts = ["kandinsky painting"]

height = 512
width = 512
num_inference_steps = 70
guidance_scale = 7.5
batch_size = 1
device = 'cuda'

# pipe
pipe = StableDiffusionPipe()
pipe.to_device(device)

# diffuse
# images = diffuse(prompts, pipe)
# imgs = mk_imgs(images)
# imgs[0].save("test.jpg")
# animate
animate(prompts, pipe, rootdir='.', name='kandinsky', device='cuda', max_frames=1000, num_steps=200, quality=90)
