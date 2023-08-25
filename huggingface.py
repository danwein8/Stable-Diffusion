# RUN THESE COMMANDS IN THE COMMAND LINE
# pip install diffusers==0.10.2 transformers scipy ftfy accelerate
# huggingface-cli login
# MAKE SURE GIT LFS IS INSTALLED
# git lfs install

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


def image_grid(imgs, rows=2, cols=2):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


pipeline = StableDiffusionPipeline.from_pretrained("BirdL/NGA_Art_SD-V1.5")
# prompt = "a photo of a Husky on a mountian top"
# image = pipeline(prompt).images[0]
# print(image)
num_images = 3
prompt = ["a painting of the founding fathers having a party"] * num_images
images = pipeline(prompt).images
grid = image_grid(images, rows=1, cols=3)
grid.save('grid.png')
