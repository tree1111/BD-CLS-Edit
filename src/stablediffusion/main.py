from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DDIMScheduler
import os
import json
import torch
import argparse
import matplotlib.pyplot as plt
from src.ds.images_utils import load_image, decode
from src.stablediffusion.bdclsedit import bdclsedit


parser = argparse.ArgumentParser()
parser.add_argument("--device-num", type=int, default=1)

parser.add_argument("--image-root-path", default='dat/img/RealWorld/')
parser.add_argument("--out-path", default='out/realworld/')
parser.add_argument("--initial-image", default='sunny.png')

parser.add_argument("--num-diffusion-steps", type=int, default=200)
parser.add_argument("--cfg-scale", type=float, default=7.5)
parser.add_argument("--skip", type=int, default=0.2)
parser.add_argument("--opt-iter", type=int, default=5)

args = parser.parse_args()

prompt_file = os.path.join("dat", "prompts", os.path.splitext(args.initial_image)[0] + ".json")
with open(prompt_file) as f:
    prompts = json.load(f)
source_prompt = prompts["source"]
target_prompt = prompts["target"]

device = f"cuda:{args.device_num}"

pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.unet.config.addition_embed_type = None

num_diffusion_steps = args.num_diffusion_steps
pipe.to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(num_diffusion_steps)

source_path = os.path.join(args.image_root_path, args.initial_image)
image = load_image(source_path, size=1024)
image_source = torch.from_numpy(image).float().permute(2, 0, 1) / 127.5 - 1
image_source = image_source.unsqueeze(0).to(device)

cfg_scale = args.cfg_scale

i0 = bdclsedit(
    pipe, source_prompt, target_prompt, image_source, 
    cfg_scale=cfg_scale, num_inference_steps=num_diffusion_steps, skip=args.skip, opt_iter=args.opt_iter)

out = decode(i0, pipe, im_cat=image)
os.makedirs(args.out_path, exist_ok=True)
out.save(f"{args.out_path}/{os.path.splitext(args.initial_image)[0]}_ctf.png")