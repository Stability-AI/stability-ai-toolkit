# stability-ai-toolkit
A collection of code samples for working with Stability AI's models. This repo will be used for technical assets that accompany blog posts on https://stability.ai/learning-hub

![Image-to-Image](./images/screenshot_image_to_image.png)

![Inpainting](./images/screenshot_inpainting.png)

## Stable Diffusion 3.5 Inference Speeds
|Model|Inference Speed (seconds)|GPU / CPU|
|-----|-------------------------|---|
|SD3.5 M|4 s|NVIDIA H100 GPU with 80 GB of VRAM|
|[4-Bit Quanitized SD3.5 L](/sd35-text-to-image-quantized-gradio/)|18 s|NVIDIA H100 GPU with 80 GB of VRAM <br> Inference partially offloaded to AWS EC2 p5.48xlarge instance's CPU: AMD EPYC 7R13|
|SD3.5 L|7 s|NVIDIA H100 GPU with 80 GB of VRAM|

## Stable Diffusion 3.5 Negative Prompting

**TL:DR; The key to removing objects isn't negative prompting but positive prompting for object placement**

* A good test of negative prompting is object removal; for example (model [4-Bit Quanitized SD3.5 L](/sd35-text-to-image-quantized-gradio/)):

  `prompt`:  Children's birthday party

  `negative_prompt`: No birthday cake

  ![NF4 SD3.5 L guidance_scale=10](./images/negative-prompting-examples/nf4%20sd3.5%20L%20guidance_scale=10.png)

* Quantization reduces the precision of the model's weights from 32-bit floating point to 4-bit floating point
* This reduction in precision makes negative prompting more effective
* For the base model of [Stable Diffusion 3.5 Large](./sd35-text-to-image-gradio/) (with no quantization or modifications) including the [API](https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post), negative prompting actually works extremely well; for example:

  `prompt`: A group of elves hunting a dragon, 4k cinema

  `negative_prompt`: No green grass

  ![SD3.5 L guidance_scale=2.5](./images/negative-prompting-examples/sd3.5%20L%20guidance_scale=2.5.png)

* For the base model of [Stable Diffusion 3.5 Large](./sd35-text-to-image-gradio/), negative prompting of specific objects (like a birthday cake) is highly dependent on prompt structure and guidance scale; for example:

  `prompt`:
  ```
  Three children sitting at a dining table
  There is a white table cloth on the table
  There are balloons in the background
  The kids are wearing party hats
  The background is a sunny day at a park
  ```

  `negative_prompt`: [blank]

  `guidance_scale`: `7.5`

  ![SD3.5 L guidance_scale=7.5](./images/negative-prompting-examples/sd3.5%20L%20guidance_scale=7.5.png)

* The key to removing objects isn't negative prompting but **positive prompting for object placement**
* This is explained in the [Stable Diffusion 3.5 Prompt Guide](https://stability.ai/learning-hub/stable-diffusion-3-5-prompt-guide)

## Stable Diffusion 3.5 Prompt Tuning Using Guidance Scale
The [guidance_scale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.guidance_scale) parameter has a significant impact on image generation with Stable Diffusion 3.5 models:
> A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality

Image quality can vary drastically based on the `guidance_scale` value. The below screenshots provide some recommended `guidance_scale` settings for three Stable Diffusion 3.5 models:
* [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) (SD3.5 L)
  * [Sample code](./sd35-text-to-image-gradio/app.py)
* [4-Bit Quantized Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) (NF4 SD3.5 L)
  * NF4: [Normal Floating Point 4](https://huggingface.co/docs/diffusers/v0.32.2/en/quantization/bitsandbytes#normal-float-4-nf4)
  * [Sample code](./sd35-text-to-image-quantized-gradio/app.py)
* [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) (SD3.5 M)

### Guidance Scale Examples
|Model|[guidance_scale](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.guidance_scale) (float 1-10)|Example|
|-----|--------------|-------|
|SD3.5 L|`guidance_scale=2.5`|![sd3.5 L guidance_scale=2.5](./images/guidance-scale-examples/sd3.5%20L%20guidance_scale=2.5.png)|
|NF4 SD3.5 L|`guidance_scale=7.5`|![nf4 sd3.5 L guidance_scale=7.5](./images/guidance-scale-examples/nf4%20sd3.5%20L%20guidance_scale=7.5.png)|
|SD3.5 M|`guidance_scale=5.0`|![sd3.5 M guidance_scale=5](./images/guidance-scale-examples/sd3.5%20M%20guidance_scale=5.png)|