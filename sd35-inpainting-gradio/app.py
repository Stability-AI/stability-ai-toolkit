# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This app makes a simple inpainting UI for Stable Diffusion
"""

import gradio as gr
import torch
import os

from diffusers import StableDiffusion3InpaintPipeline
from huggingface_hub import login
from PIL import Image

class StableUI:
    _pipe = []

    def __init__(self):
        pass

    def _check_shader(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        return device

    def _predict(self, mask, prompt):

        # Extract the image and mask channels
        image = mask['background'].convert("RGB")
        mask = mask['layers'][0]
        white_background = Image.new('RGB', (512, 512), (255, 255, 255))
        mask_image = white_background.resize(mask.size)
        mask_image.paste(mask, (0, 0), mask)

        # White pixels in the mask are repainted while black pixels are preserved, documentation:
        #   https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint#diffusers.StableDiffusionInpaintPipeline.__call__.mask_image
        image.show()
        mask_image.show()

        images = self._pipe(prompt=prompt, image=image, mask_image=mask_image).images
        return images[0]

    def _start_gradio(self):
        gr.Interface(
            self._predict,
            title='Stable Diffusion 3.5 Large In-Painting',
            inputs=[
                gr.ImageMask(type='pil', label='Inpaint', canvas_size=(716, 716)),
                gr.Textbox(label='prompt')
            ],
            outputs='image'
        ).launch(debug=True, share=True)

    def start_inpaint(self): 
        # Make sure the HUGGING_FACE_HUB_TOKEN environment variable is set
        # with your Hugging Face Access token, or log in from the command line:
        #
        #  huggingface-cli login
        #
        # Instructions for getting an access token: https://huggingface.co/docs/hub/en/security-tokens
        if os.getenv('HUGGING_FACE_HUB_TOKEN') or os.getenv('HF_TOKEN'):
            print("Hugging Face access token set")
        else:
            login()

        self._pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
        device = self._check_shader()
        self._pipe.to(device)

        self._start_gradio()
        return 0

def main():
    ui = StableUI()
    ui.start_inpaint()

if __name__ == "__main__":
    main()