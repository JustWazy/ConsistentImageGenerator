import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import gc

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    enable_model_cpu_offload=True,  
    force_zeros_for_empty_prompt=False
).to("cuda")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    refiner_model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    enable_model_cpu_offload=True
).to("cuda")

def high_quality_generate(
    prompt,
    negative_prompt="",
    ref_image=None,
    strength=0.45,
    width=896,  
    height=896,
    base_steps=30,
    refine_steps=15
):
  
    torch.cuda.empty_cache()
    
    if ref_image:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=ref_image,
            strength=strength,
            num_inference_steps=base_steps,
            guidance_scale=8.5,
            width=width,
            height=height,
            original_size=(1024, 1024),
            target_size=(width, height),
            use_karras_sigmas=True, 
            output_type="latent" 
        ).images[0]
    else:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=base_steps,
            guidance_scale=8.5,
            width=width,
            height=height,
            original_size=(1024, 1024),
            target_size=(width, height),
            use_karras_sigmas=True,
            output_type="latent"
        ).images[0]

    refined_image = refiner(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=refine_steps,
        strength=0.3,
        guidance_scale=5.0,
        original_size=(1024, 1024),
        target_size=(width, height)
    ).images[0]

    del image
    gc.collect()
    torch.cuda.empty_cache()
    
    return refined_image

def main():
    last_output = None
    
    while True:
        user_prompt = input("\nPrompt (atau 'exit'): ")
        if user_prompt.lower() == 'exit':
            break
            
        neg_prompt = input("Negative prompt (enter untuk skip): ")
        
        use_ref = input("Gunakan referensi gambar sebelumnya? (y/n): ").lower()
        ref_image = last_output if use_ref == 'y' else None
        
        try:
            custom_size = input("Ukuran (lebar tinggi contoh: 896 1152): ")
            if custom_size:
                width, height = map(int, custom_size.split())
            else:
                width, height = 896, 896
        except:
            width, height = 896, 896
        
        try:
            output = high_quality_generate(
                prompt=user_prompt,
                negative_prompt=neg_prompt,
                ref_image=ref_image,
                width=width,
                height=height
            )
            display(output)
            output.save("hq_output.jpg")
            last_output = output
        except Exception as e:
            print(f"Error: {e}")
            pipe = None
            refiner = None
            torch.cuda.empty_cache()
            pipe = StableDiffusionXLPipeline.from_pretrained(...)
            refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(...)

if __name__ == "__main__":
    main()
