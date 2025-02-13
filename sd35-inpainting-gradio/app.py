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

    # ... (previous code remains the same)

    def _predict(self, mask, strength, guidance_scale, prompt, negative_prompt, progress=gr.Progress(track_tqdm=True)):
        image = mask['background'].convert("RGB")
        mask_image = mask['layers'][0].convert("L")
        mask_image = mask_image.resize(image.size)
        
        width, height = image.size
    
        # Calculate new dimensions preserving aspect ratio
        min_dim = min(width, height)
        
        # First scale to minimum 512px on smallest side
        if min_dim < 512:
            scale_factor = 512 / min_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = width
            new_height = height
    
        # Now ensure dimensions are multiples of 64
        new_width = (new_width // 64) * 64
        new_height = (new_height // 64) * 64
    
        # Apply final resizing
        image = image.resize((new_width, new_height))
        mask_image = mask_image.resize((new_width, new_height))
    
        images = self._pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            width=new_width,
            height=new_height,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt
        ).images
        return images[0]


    def _start_gradio(self):
        white_brush = gr.Brush(default_color='#FFFFFF', colors=['#FFFFFF'], color_mode='fixed')

        gr.Interface(
            self._predict,
            title='Stable Diffusion 3.5 Large In-Painting',
            inputs=[
                gr.ImageMask(type='pil', label='Inpaint', height="680", brush=white_brush),
                gr.Slider(minimum=0, maximum=1, value=1.0, label="strength (increase inpainting strength)"),
                gr.Slider(minimum=1, maximum=10, value=7.5, label="guidance scale (increase to apply text prompt)"),
                gr.Textbox(label='prompt'),
                gr.Textbox(label='negative prompt')
            ],
            outputs=gr.Image(type="pil")
        ).launch(debug=True, share=True)

    def start_inpaint(self):
        self._pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
        device = self._check_shader()
        self._pipe.to(device)

        self._start_gradio()
        return 0

def main():
    ui = StableUI()
    ui.login_to_hugging_face()
    ui.start_inpaint()

if __name__ == "__main__":
    main()