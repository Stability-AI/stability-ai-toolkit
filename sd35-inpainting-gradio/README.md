# Stable Diffusion 3.5 Inpainting in Gradio
Gradio demo of inpainting using Stable Diffusion 3.5 Large

### Screenshot
![screenshot.png](./images/screenshot.png)

#### Input Image and Gradio ImageMask
![example_input_256x256.png](./images/example_input_256x256.png) ![](./images/example_mask_256x256.png)

## Quick Start
1. Open a web browser, log in to Hugging Face and register your name and email,
   to use [stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
2. Create a new Hugging Face [user access token](https://huggingface.co/docs/hub/en/security-tokens),
   which will capture that you completed the registration form
3. Clone this repo to your machine and change into the directory for this demo:
   ```
   cd ./stability-ai-toolkit/sd35-inpainting-gradio
   ```
4. Set up the app in a Python virtual environment:

   ```
   python -m venv <your_environment_name>
   source <your_environment_name>/bin/activate
   ```
5. Set your `HF_TOKEN` inside your virtual environment
   ```
   export HF_TOKEN=<Hugging Face user access token>
   ```
6. Install dependencies
   ```
   pip install -r requirements.txt
   ```

   NOTE: Read [requirements.txt](./requirements.txt) for
   [MacOS PyTorch installation instructions](https://developer.apple.com/metal/pytorch/)

   TL;DR:
   ```
   # Inside your virtual environment
   pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```
7. Start the app
   ```
   python app.py
   ```
8. Open UI in a web browser: [http://127.0.0.1:7861](http://127.0.0.1:7861)

## Usage Tips
* If you're trying to modify the code into your own app, swap out `stabilityai/stable-diffusion-3.5-large` for
  `stabilityai/stable-diffusion-3.5-medium`, which has faster inference time (but lower quality)

  Once you're finished development then swap `stabilityai/stable-diffusion-3.5-large` for higher quality output