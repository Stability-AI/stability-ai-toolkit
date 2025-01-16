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
This app makes a simple image-generation UI for Stable Diffusion
"""

import gradio as gr
import torch
import os

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
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

    def _predict(self, guidance_scale, prompt, negative_prompt, progress=gr.Progress(track_tqdm=True)):
        images = self._pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt
        ).images

        return images[0]

    def _start_gradio(self):
        gr.Interface(
            self._predict,
            title='4-Bit Quantized Stable Diffusion 3.5 Large Text-to-Image',
            inputs=[
                gr.Slider(minimum=1, maximum=10, value=7.5, label="guidance scale (increase to apply text prompt)"),
                gr.Textbox(label='prompt'),
                gr.Textbox(label='negative prompt')
            ],
            outputs='image'
        ).launch(debug=True, share=True)

    def start_text_to_image(self):
        model_id = "stabilityai/stable-diffusion-3.5-large"

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        self._pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        self._pipe.enable_model_cpu_offload()

        self._start_gradio()
        return 0

def main():
    ui = StableUI()
    ui.login_to_hugging_face()
    ui.start_text_to_image()

if __name__ == "__main__":
    main()