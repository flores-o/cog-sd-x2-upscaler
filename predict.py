import os
from typing import List
from PIL import Image

import torch
from cog import BasePredictor, Input, Path, File
from diffusers import (
    StableDiffusionLatentUpscalePipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


MODEL_ID = "stabilityai/sd-x2-latent-upscaler"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        image: File = Input(description="Input image to upscale"),
        prompt: str = Input(
            description="Input prompt",
            default="",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="K_EULER",
            choices=[
                "K_EULER",
                "K_EULER_ANCESTRAL",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        # print(f"Using seed: {seed}")

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        #     )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)


        # Measure the memory usage before inference
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # in MB

        device = torch.device("cuda")
        print(f"====================================================================================")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"====================================================================================")



        # Memory Efficient Attention
        self.pipe.enable_xformers_memory_efficient_attention()

        import time
        start_time = time.time()

        output = self.pipe(
            prompt=prompt,
            image=Image.open(image),
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency:.4f} seconds")
        

        # Measure the CPU memory usage after inference
        mem_after = process.memory_info().rss / 1024 / 1024  # in MB
        mem_used = mem_after - mem_before
        
        print(f"====================================================================================")
        print(f"CPU Memory used: {mem_used:.2f} MB")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")
        print(f"====================================================================================")

        output_paths = []
        for i, sample in enumerate(output.images):
            # if output.nsfw_content_detected[i]:
            #     continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        # if len(output_paths) == 0:
        #     raise Exception(
        #         f"NSFW content detected. Try running it again, or try a different prompt."
        #     )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
