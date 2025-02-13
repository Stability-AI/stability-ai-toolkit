# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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

torch.mps.empty_cache()
class StableUI:
    _pipe = []

    def __init__(self):
        pass

    def login_to_hugging_face(self):
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
            print("\nWARNING: To avoid the Hugging Face login prompt in the future, please set the HF_TOKEN environment variable:\n\n    export HF_TOKEN=<YOUR HUGGING FACE USER ACCESS TOKEN>\n")

    def _check_shader(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        return device

    def _predict(self, mask, prompt, progress=gr.Progress(track_tqdm=True)):

        # Extract the image and mask channels
        image = mask['background'].convert("RGB")
        mask_image = mask['layers'][0].convert("RGB")

        image.show()
        mask_image.show()

        images = self._pipe(prompt=prompt, image=image, mask_image=mask_image).images
        return images[0]

    def _start_gradio(self):
        white_brush = gr.Brush(colors=['#FFFFFF'], color_mode='fixed')

        gr.Interface(
            self._predict,
            title='Stable Diffusion 3.5 Large In-Painting',
            inputs=[
                gr.ImageMask(type='pil', label='Inpaint', height="680", brush=white_brush),
                gr.Textbox(label='prompt')
            ],
            outputs='image'
        ).launch(debug=True, share=True)

    def start_inpaint(self):
        self._pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16)
        device = self._check_shader()
        self._pipe.to(device, torch.float16)

        self._start_gradio()
        return 0

def main():
    ui = StableUI()
    ui.login_to_hugging_face()
    ui.start_inpaint()

if __name__ == "__main__":
    main()
