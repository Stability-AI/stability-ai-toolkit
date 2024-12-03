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

from diffusers import AutoPipelineForImage2Image
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

    def _predict(self, init_image, strength, guidance_scale, prompt, negative_prompt, progress=gr.Progress(track_tqdm=True)):
        init_image = init_image.resize((1024, 1024))

        images = self._pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt
        ).images

        return images[0]

    def _start_gradio(self):
        gr.Interface(
            self._predict,
            title='Stable Diffusion 3.5 Medium Image-to-Image',
            inputs=[
                gr.Image(type='pil', label='Initial Image'),
                gr.Slider(minimum=0, maximum=1, value=0.75, label="strength (increase to ignore input image)"),
                gr.Slider(minimum=1, maximum=10, value=7.5, label="guidance scale (increase to apply text prompt)"),
                gr.Textbox(label='prompt'),
                gr.Textbox(label='negative prompt')
            ],
            outputs='image'
        ).launch(debug=True, share=True)

    def start_image_to_image(self):
        self._pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16
        )
        device = self._check_shader()
        self._pipe.to(device)

        self._start_gradio()
        return 0

def main():
    ui = StableUI()
    ui.login_to_hugging_face()
    ui.start_image_to_image()

if __name__ == "__main__":
    main()