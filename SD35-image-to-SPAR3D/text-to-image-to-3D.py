import gradio as gr
from io import BytesIO
import json
from PIL import Image
import requests
import os

STABILITY_KEY = None
IMAGE_PROMPT = None

def save_key_and_generate_image(key, prompt):
    global STABILITY_KEY, IMAGE_PROMPT
    STABILITY_KEY = key
    IMAGE_PROMPT = prompt
    print(f"Stability KEY is : {STABILITY_KEY}")
    print(f"Image Prompt is : {IMAGE_PROMPT}")

    image = generate_image()
    image_path = "./generated_image.png"
    image.save(image_path)
    print(f"Image saved to {image_path}")

    object_path = generate_3d_object(image_path)
    return image_path, object_path

def send_generation_request(host, params):
    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "image/*"
    }
    response = requests.post(host, headers=headers, files=params)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e}")
        print(f"Response content: {response.content}")
        raise
    return response

def generate_image():
    prompt = IMAGE_PROMPT
    negative_prompt = ""
    aspect_ratio = "1:1"
    seed = 0
    output_format = "jpeg"

    host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"

    params = {
        "prompt": (None, prompt),
        "negative_prompt": (None, negative_prompt),
        "aspect_ratio": (None, aspect_ratio),
        "seed": (None, str(seed)),
        "output_format": (None, output_format),
        "model": (None, "sd3.5-large"),
        "mode": (None, "text-to-image")
    }

    print(f"Sending request with params: {json.dumps(params, indent=2)}")
    response = send_generation_request(host, params)

    # Decode response
    output_image = Image.open(BytesIO(response.content))
    finish_reason = response.headers.get("finish-reason")
    seed = response.headers.get("seed")

    # Check for NSFW classification
    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")

    return output_image

def generate_3d_object(image_path):
    response = requests.post(
        f"https://api.stability.ai/v2beta/3d/stable-point-aware-3d",
        headers={
            "authorization": f"Bearer {STABILITY_KEY}",
        },
        files={
            "image": open(image_path, "rb")
        },
        data={},
    )

    if response.status_code == 200:
        output_3d_path = "./3d_object.glb"
        with open(output_3d_path, 'wb') as file:
            file.write(response.content)
        print(f"3D object saved to {output_3d_path}")
        return output_3d_path
    else:
        raise Exception(str(response.json()))

demo = gr.Interface(
    fn=save_key_and_generate_image,
    inputs=[gr.Textbox(label="Enter your StabilityAI Key", type="password"), gr.Textbox(label="Enter your Image Prompt")],
    outputs=[gr.Image(label="Generated Image"), gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model")],
    title="Stability AI | Text to Image to 3D"
)

if __name__ == "__main__":
    demo.launch()