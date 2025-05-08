import gradio as gr
import time
import torch
import gc
import random
from PIL import Image # For image concatenation
# from text_to_image_generation import generate_images # Not strictly needed if run_inference is self-contained
from optimum.habana.diffusers import GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline
from optimum.habana.utils import set_seed # For setting the seed

MODEL_PATH = "/workspace/models/FLUX.1-dev"
IMG_SAVE_PATH = "/workspace/jh/flux/outputs/gradio"
RDT = 0.2
HEIGHT = 512
WIDTH = 512
NUM_IMAGES_PER_PROMPT = 4 # Define as a constant for clarity
BATCH_SIZE = 4 # Define as a constant for clarity

SAMPLE_PROMPTS = [
    "Hyper-real 3-D render of a transparent, high-gloss glass toy brick with four rounded studs on top. Smooth beveled edges shimmer in a cobalt-to-magenta gradient. Sharp specular highlights and a subtle inner glow give depth. The brick floats against pure black, emphasizing sheen and color.",
    "Hyper-real ray-traced 3-D render: slender, transparent glass rocket icon rotated 45° right. Smooth conical nose, tapered delta fins, rounded edges. Emerald-teal-amber edge gradient, sharp specular highlights, cyan core glow. Round window on fuselage, vivid red exhaust flame. Pure black background.",
    "Storybook-style digital illustration of a rainy spring evening in Seoul. Neon Hangul signs shimmer on wet asphalt; cherry petals drift through fog as commuters with umbrellas pass glowing shops. Painterly gouache-watercolor texture, pastel-neon palette, soft diffusion, light film grain, nostalgic yet modern.",
    "Ultra-sharp 50 mm f/1.4 portrait of a Korean woman, late-20s, beside sheer-curtained window. Soft side light forms a Rembrandt triangle, revealing warm skin and crisp hair strands. Neutral linen blouse, calm genuine gaze. Wide-open aperture for creamy bokeh, true color, editorial realism"
]

# 전역 변수로 파이프라인 관리
pipeline = None

# 파이프라인 초기화 함수
def initialize_pipeline():
    global pipeline
    
    if pipeline is None:
        print("Initializing pipeline for the first time...")
        scheduler_obj = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_PATH, subfolder="scheduler", timestep_spacing="linspace"
        )
        kwargs_pipe = {
            "use_habana": True,
            "gaudi_config": "Habana/stable-diffusion",
            "sdp_on_bf16": True,
            "scheduler": scheduler_obj,
            "torch_dtype": torch.bfloat16
        }
        pipeline = GaudiFluxPipeline.from_pretrained(MODEL_PATH, **kwargs_pipe)
        if True:  # use_fbcache
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            apply_cache_on_pipe(pipeline, residual_diff_threshold=RDT)
    return pipeline

# 추론 함수
def run_inference(prompt):
    pipe = initialize_pipeline()
    
    current_seed = random.randint(0, 2**32 - 1)
    print(f"Using seed for this run: {current_seed}")
    set_seed(current_seed) 
        
    kwargs_call = {
        "prompt": prompt,
        "num_images_per_prompt": NUM_IMAGES_PER_PROMPT,
        "batch_size": BATCH_SIZE,
        "num_inference_steps": 28,
        "output_type": "pil",
        "height": HEIGHT,
        "width": WIDTH,
        "throughput_warmup_steps": 0,
    }
    
    t0 = time.time()
    outputs = pipe(**kwargs_call)
    t1 = time.time()
    
    images = outputs.images if hasattr(outputs, "images") else outputs["images"]
    elapsed = t1 - t0
    
    if IMG_SAVE_PATH is not None:
        import os
        from pathlib import Path
        image_save_dir = Path(IMG_SAVE_PATH)
        image_save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        if not isinstance(images, list):
            images_list = [images]
        else:
            images_list = images
        for i, image in enumerate(images_list):
            try:
                image.save(image_save_dir / f"image_{timestamp}_{current_seed}_{i + 1}.png")
            except AttributeError:
                 print(f"Warning: Could not save image {i+1} as it might not be a PIL image.")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'hpu') and torch.hpu.is_available():
        pass 
            
    return images, elapsed

def concat_images_grid(images, cols=2):
    if not images or not isinstance(images, list):
        return None
    if len(images) != 4: # Specifically for 2x2 grid
        print(f"Warning: Expected 4 images for 2x2 grid, got {len(images)}. Returning list.")
        return images # Return original list if not 4 images

    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# Custom CSS for gallery image scaling
# This CSS targets images within the Gradio gallery. 
# It might need to be more specific if it affects other images on the page.
gallery_image_scale_css = """
.gradio-container .gallery-item img {
    transform: scale(2.0) !important;
    transform-origin: center center !important; /* Ensures scaling is from the center */
    object-fit: contain !important; /* Adjust as needed: contain, cover, fill, etc. */
}
.gallery-item {
    overflow: visible !important; /* Allow scaled image to overflow item boundaries */
}
"""

with gr.Blocks(title="Flux[dev] model inference on Gaudi-2 HPU", css=gallery_image_scale_css) as demo:
    gr.Markdown("# Flux[dev] model inference on Gaudi-2 HPU")
    with gr.Row():
        with gr.Column():
            left_time = gr.Markdown("Elapsed: ... seconds")
            left_icon_display = gr.HTML(visible=False)
            left_gallery = gr.Gallery(label="Immediate Result", visible=True)
        with gr.Column():
            right_time = gr.Markdown("Elapsed: ... seconds")
            right_icon_display = gr.HTML(visible=False)
            right_gallery = gr.Gallery(label="Delayed Result (2-4s)", visible=True)
    
    prompt_box = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=2)
    
    with gr.Row():
        sample_prompt_btn = gr.Button("Use Sample Prompt")
        submit_btn = gr.Button("Generate", variant="primary")

    cheetah_icon_html = "<div style='font-size:50px; text-align:center;'>🐆</div>"
    turtle_icon_html = "<div style='font-size:50px; text-align:center;'>🐢</div>"

    def on_submit_generate(prompt):
        loading_msg = "⏳ Generating images... Please wait."
        yield (
            gr.update(value=loading_msg),                                  
            gr.update(visible=True, value=cheetah_icon_html), 
            gr.update(visible=False, value=None),                          
            gr.update(value=loading_msg),                                  
            gr.update(visible=True, value=turtle_icon_html),               
            gr.update(visible=False, value=None)                           
        )
        
        images, elapsed = run_inference(prompt)
        display_images = images 
        if NUM_IMAGES_PER_PROMPT == 4 and isinstance(images, list) and len(images) == 4:
            concatenated_image = concat_images_grid(images)
            if concatenated_image:
                display_images = [concatenated_image] 
            left_time_text = f"{NUM_IMAGES_PER_PROMPT} images are generated in {elapsed:.2f} seconds <br>"
            left_time_text += f"1 image is generated in {elapsed/NUM_IMAGES_PER_PROMPT:.2f} seconds <br>"
        else:
            left_time_text = f"1 image is generated in {elapsed:.2f} seconds <br>"
        
        delay_time = random.uniform(2.0, 4.0)
        right_loading_msg = f"⏳ Will be displayed soon..."

        yield (
            gr.update(value=left_time_text),
            gr.update(visible=False),                                      
            gr.update(visible=True, value=display_images),                
            gr.update(value=right_loading_msg),
            gr.update(visible=True, value=turtle_icon_html),               
            gr.update(visible=False, value=None)                           
        )
        
        time.sleep(delay_time)
        
        total_elapsed_right = elapsed + delay_time
        if NUM_IMAGES_PER_PROMPT == 4 and isinstance(images, list) and len(images) == 4:
            right_time_text = f"{NUM_IMAGES_PER_PROMPT} images are generated in {elapsed:.2f} seconds (includes {delay_time:.2f}s delay) <br>"
            right_time_text += f"1 image is generated in {elapsed/NUM_IMAGES_PER_PROMPT:.2f} seconds (includes {delay_time/NUM_IMAGES_PER_PROMPT:.2f}s delay) <br>"
        else:
            right_time_text = f"Elapsed: {total_elapsed_right:.2f} seconds (includes {delay_time:.2f}s delay) <br>"
        
        yield (
            gr.update(value=left_time_text),                               
            gr.update(visible=False),                                      
            gr.update(visible=True, value=display_images),                
            gr.update(value=right_time_text),
            gr.update(visible=False),                                      
            gr.update(visible=True, value=display_images)                         
        )

    def on_click_sample_prompt():
        return gr.update(value=random.choice(SAMPLE_PROMPTS))

    submit_btn.click(
        on_submit_generate,
        inputs=prompt_box,
        outputs=[left_time, left_icon_display, left_gallery, right_time, right_icon_display, right_gallery],
    )
    prompt_box.submit(
        on_submit_generate,
        inputs=prompt_box,
        outputs=[left_time, left_icon_display, left_gallery, right_time, right_icon_display, right_gallery],
    )
    sample_prompt_btn.click(fn=on_click_sample_prompt, inputs=None, outputs=prompt_box)

demo.launch(share=True) 