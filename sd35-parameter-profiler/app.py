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
This app makes gets the parameters (weights) used in a Stable Diffusion 3.5 model
"""

from safetensors.torch import load_file
import os

def main():
    # Load the safetensors file
    model_path = f"{os.getenv('MODEL_PATH')}/sd3.5_large.safetensors"
    state_dict = load_file(model_path)

    # Count total parameters
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    print(f"Total Parameters: {total_params}")

if __name__ == "__main__":
    main()