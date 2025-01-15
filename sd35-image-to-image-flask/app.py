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
This app exposes a REST endpoint for Stable Diffusion image-to-image
"""

import os
import sys
import torch

from diffusers import AutoPipelineForImage2Image
from flask import Flask, request, send_file
from PIL import Image

app = Flask(__name__)
model_path = os.environ.get('MODEL_PATH')
try:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
except OSError as e:
    print("\nPlease set environment variable MODEL_PATH equal to the absolute path of your downloaded Stable Diffusion model, for example:\n\nexport MODEL_PATH=/absolute/path/to/stable-diffusion-3.5-large\n", file=sys.stderr)
    sys.exit(1)

def check_shader():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    init_image = Image.open(image_file.stream).resize((1024, 1024))

    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidance_scale'])

    images = pipe(
        prompt=request.form['prompt'],
        image=init_image,
        strength=strength,
        guidance_scale=guidance_scale,
        negative_prompt=request.form['negative_prompt']
    ).images

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = f'{script_dir}/images/output.png'
    images[0].save(image_path, 'PNG')

    return send_file(image_path, mimetype='image/png')

def main():
    device = check_shader()
    pipe.to(device)

    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    return 0

if __name__ == "__main__":
    main()