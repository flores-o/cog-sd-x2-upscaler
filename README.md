# Stable Diffusion Upscaler Cog model



This is an implementation of the [Stable Diffusion x2 latent upscaler](https://huggingface.co/stabilityai/sd-x2-latent-upscaler) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

    cog predict -i image=@input_image.png

