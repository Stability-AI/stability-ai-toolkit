# Stable Diffusion 3.5 Parameter Profiler
This repo folder is for profiling Stable Diffusion 3.5 model parameters

# Quick Start

**NOTE:** In this code sample the Stable Diffusion 3.5 model is loaded locally using environment variable `MODEL_PATH` and not from Hugging Face

1. Open a web browser, log in to Hugging Face and register your name and email
2. git clone [stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) (or your desired Stable Diffusion 3.5 version) to your local machine
   ```
   git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-large
   ```
3. Clone this repo ([stability-ai-toolkit](../)) to your local machine and change into the directory for this demo:
   ```
   cd ./stability-ai-toolkit/sd35-parameter-profiler
   ```
4. Create a Python 3.10 virtual environment:
   ```
   python3.10 -m venv <your_environment_name>
   source <your_environment_name>/bin/activate
   ```
5. Inside your virutal environment, set the `MODEL_PATH` environment variable equal to the absolute path of your Stable Diffusion model downloaded in step 2 above, for example:
   ```
   export MODEL_PATH=/absolute/path/to/stable-diffusion-3.5-large
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
7. Start the Flask app
   ```
   python app.py
   ```
   Output
   ```
   Total Parameters: 8230100451
   ```