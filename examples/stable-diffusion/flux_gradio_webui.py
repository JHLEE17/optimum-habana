import gradio as gr
import time
import torch
import gc
import random
from PIL import Image # For image concatenation
from optimum.habana.diffusers import GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline
from optimum.habana.utils import set_seed # For setting the seed

MODEL_PATH = "/workspace/models/FLUX.1-dev"
IMG_SAVE_PATH = "/workspace/jh/flux/outputs/gradio"
RDT = 0.2
HEIGHT = 1024
WIDTH = 1024
NUM_IMAGES_PER_PROMPT = 1 # Define as a constant for clarity
BATCH_SIZE = 1 # Define as a constant for clarity
FP8 = False
USE_HPU_GRAPHS = True
RESIZE_IMAGES = False # Option to disable resizing

SAMPLE_PROMPTS = [
    "Hyper-real 3-D render of a transparent, high-gloss glass toy brick with four rounded studs on top. Smooth beveled edges shimmer in a cobalt-to-magenta gradient. Sharp specular highlights and a subtle inner glow give depth. The brick floats against pure black, emphasizing sheen and color.",
    "Hyper-real ray-traced 3-D render: slender, transparent glass rocket icon rotated 45° right. Smooth conical nose, tapered delta fins, rounded edges. Emerald-teal-amber edge gradient, sharp specular highlights, cyan core glow. Round window on fuselage, vivid red exhaust flame. Pure black background.",
    # "Storybook-style digital illustration of a rainy spring evening in Seoul. Neon Hangul signs shimmer on wet asphalt; cherry petals drift through fog as commuters with umbrellas pass glowing shops. Painterly gouache-watercolor texture, pastel-neon palette, soft diffusion, light film grain, nostalgic yet modern.",
    "Ultra-sharp 50 mm f/1.4 portrait of a Korean woman, late-20s, beside sheer-curtained window. Soft side light forms a Rembrandt triangle, revealing warm skin and crisp hair strands. Neutral linen blouse, calm genuine gaze. Wide-open aperture for creamy bokeh, true color, editorial realism",
    "Ultra-high-resolution product photo of a luxury mechanical wristwatch lying on a brushed-steel surface, sapphire crystal facing the camera at 45 °, showing the open tourbillon cage, engraved minute markers, and a faint fingerprint on the glass. Studio lighting with multiple specular highlights and soft-box rim light.",
    "Minimalist studio still-life of a clear glass teapot filled with amber oolong tea on a pure-white marble slab. Soft diagonal daylight, subtle caustics on the surface, gentle shadow fall-off, no other objects in frame."
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
            "use_hpu_graphs": USE_HPU_GRAPHS,
            "gaudi_config": "Habana/stable-diffusion",
            "sdp_on_bf16": True,
            "scheduler": scheduler_obj,
            "torch_dtype": torch.bfloat16,
            "rdt": RDT
        }
        pipeline = GaudiFluxPipeline.from_pretrained(MODEL_PATH, **kwargs_pipe)
        if RDT > 0:  # use_fbcache
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
        "quant_mode": "quantize" if FP8 else None,
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

# 이미지를 직접 리사이즈하는 함수 (5배 크기로 조정 또는 원본 유지)
def resize_image(img, scale_factor=5.0):
    if img is None or not RESIZE_IMAGES:
        return img
    
    # PIL 이미지인지 확인
    if isinstance(img, Image.Image):
        w, h = img.size
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        print(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img

def resize_images_list(images, scale_factor=5.0):
    if not images or not RESIZE_IMAGES:
        return images
    
    if isinstance(images, list):
        return [resize_image(img, scale_factor) for img in images]
    else:
        return resize_image(images, scale_factor)

def concat_images_grid(images, cols=2):
    if not images or not isinstance(images, list):
        return None
    if len(images) != 4: # Specifically for 2x2 grid
        print(f"Warning: Expected 4 images for 2x2 grid, got {len(images)}. Returning list.")
        return images # Return original list if not 4 images

    # First resize all images to ensure they're large (only if resizing is enabled)
    large_images = images
    if RESIZE_IMAGES:
        large_images = resize_images_list(images, scale_factor=2.0)
    
    rows = (len(large_images) + cols - 1) // cols
    w, h = large_images[0].size
    
    # Create a grid with no gaps between images
    grid_width = cols * w
    grid_height = rows * h
    
    grid = Image.new('RGB', size=(grid_width, grid_height))
    
    for i, img in enumerate(large_images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    
    # Now resize the entire grid if needed
    if RESIZE_IMAGES:
        return resize_image(grid, scale_factor=2.0)
    else:
        return grid

# Custom CSS for improved layout
custom_css = """
/* Main title styling */
.main-title {
    font-size: 3em !important;
    font-weight: bold !important;
    text-align: center !important;
    margin: 10px 0 !important;
    padding: 5px !important;
    background: linear-gradient(90deg, #2d3748, #1a202c) !important;
    color: white !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

/* Title styling */
.title-text {
    margin-bottom: 10px !important;
    font-weight: bold !important;
    font-size: 1.5em !important;
}

/* Time text styling */
.time-text {
    margin-top: 10px !important;
    margin-bottom: 20px !important;
    padding: 10px !important;
    background-color: rgba(0, 0, 0, 0.1) !important;
    border-radius: 5px !important;
}

/* Full viewport height containers */
.contain-row {
    min-height: 600px !important;
}

/* Image styling - expand to fill container */
.large-image {
    width: 100% !important;
    height: 600px !important;
    object-fit: contain !important;
}

/* Force images to be large */
.large-image img {
    width: 100% !important;
    max-height: 100% !important;
}

/* Put a border around images to see them better */
.large-image img {
    border: 2px solid #444 !important;
    border-radius: 8px !important;
}

/* Adjust padding in column containers */
.gradio-column {
    padding: 10px !important;
}
"""

with gr.Blocks(title="Flux Image Generation UI", css=custom_css) as demo:
    # Main title at the top
    gr.Markdown(f"# Flux[dev] {HEIGHT}x{WIDTH} Image Generation Demo", elem_classes=["main-title"])
    
    with gr.Row(elem_classes=["contain-row"]):
        with gr.Column() as left_column:
            # Left column title
            gr.Markdown("# Intel Gaudi-2 HPU 🤝 SqueezeBits", elem_classes=["title-text"])
            
            # Replace gallery with image component for direct full-size display
            left_image = gr.Image(
                label="Immediate Result", 
                visible=False,
                elem_id="left-image",
                elem_classes=["large-image"],
                height=600,
                show_download_button=True
            )
            
            # Icons go below gallery but above time info
            left_icon_display = gr.HTML(visible=False)
            
            # Time text below gallery
            left_time = gr.HTML("Elapsed: ... seconds", elem_classes=["time-text"])
            
        with gr.Column() as right_column:
            # Right column title
            gr.Markdown("# Other API Service", elem_classes=["title-text"])
            
            # Replace gallery with image component for direct full-size display
            right_image = gr.Image(
                label="Delayed Result (2-4s)",
                visible=False,
                elem_id="right-image",
                elem_classes=["large-image"],
                height=600,
                show_download_button=True
            )
            
            # Icons go below gallery but above time info
            right_icon_display = gr.HTML(visible=False)
            
            # Time text below gallery
            right_time = gr.HTML("Elapsed: ... seconds", elem_classes=["time-text"])
    
    # Move prompt area here - immediately after the galleries
    prompt_box = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=2)
    
    with gr.Row():
        sample_prompt_btn = gr.Button("Use Sample Prompt")
        submit_btn = gr.Button("Generate", variant="primary")

    cheetah_icon_html = "<div style='font-size:50px; text-align:center;'>🐆</div>"
    turtle_icon_html = "<div style='font-size:50px; text-align:center;'>🐢</div>"

    def on_submit_generate(prompt):
        loading_msg = "<p style='text-align:center; font-weight:bold;'>⏳ Generating images... Please wait.</p>"
        yield (
            gr.update(value=loading_msg),                                  
            gr.update(visible=True, value=cheetah_icon_html), 
            gr.update(visible=False),                          
            gr.update(value=loading_msg),                                  
            gr.update(visible=True, value=turtle_icon_html),               
            gr.update(visible=False)                           
        )
        
        images, elapsed = run_inference(prompt)
        
        # Create a single large image from all generated images
        display_image = None
        if NUM_IMAGES_PER_PROMPT == 4 and isinstance(images, list) and len(images) == 4:
            # Create a 2x2 grid image
            concatenated_image = concat_images_grid(images)
            if concatenated_image:
                display_image = concatenated_image
        else:
            # If not creating a grid, just use the first image
            if isinstance(images, list) and len(images) > 0:
                display_image = images[0]
            else:
                display_image = images
        
        # HTML로 시간 텍스트 포맷팅 - 다크모드 친화적 스타일
        left_time_html = f"""
        <div style='padding:15px; background-color:#333333; color:white; border-radius:5px; font-size:1.2em;'>
            <p><b>{NUM_IMAGES_PER_PROMPT} images</b> are generated in <b>{elapsed:.2f} seconds</b></p>
            <p><b>1 image</b> is generated in <b>{elapsed/NUM_IMAGES_PER_PROMPT:.2f} seconds</b></p>
        </div>
        """
        
        delay_time = random.uniform(3.0, 4.0)
        right_loading_html = "<p style='text-align:center; font-weight:bold; font-size:1.2em;'>⏳ Will be displayed soon...</p>"

        yield (
            gr.update(value=left_time_html),
            gr.update(visible=False),                                      
            gr.update(visible=True, value=display_image),                
            gr.update(value=right_loading_html),
            gr.update(visible=True, value=turtle_icon_html),               
            gr.update(visible=False)                           
        )
        
        time.sleep(delay_time)
        
        total_elapsed_right = elapsed + delay_time
        # HTML로 시간 텍스트 포맷팅 - 다크모드 친화적 스타일
        right_time_html = f"""
        <div style='padding:15px; background-color:#333333; color:white; border-radius:5px; font-size:1.2em;'>
            <p><b>{NUM_IMAGES_PER_PROMPT} images</b> are generated in <b>{total_elapsed_right:.2f} seconds</b> (includes {delay_time:.2f}s delay)</p>
            <p><b>1 image</b> is generated in <b>{total_elapsed_right/NUM_IMAGES_PER_PROMPT:.2f} seconds</b> (includes {delay_time/NUM_IMAGES_PER_PROMPT:.2f}s delay)</p>
        </div>
        """
        
        yield (
            gr.update(value=left_time_html),                               
            gr.update(visible=False),                                      
            gr.update(visible=True, value=display_image),                
            gr.update(value=right_time_html),
            gr.update(visible=False),                                      
            gr.update(visible=True, value=display_image)                         
        )

    def on_click_sample_prompt():
        return gr.update(value=random.choice(SAMPLE_PROMPTS))

    submit_btn.click(
        on_submit_generate,
        inputs=prompt_box,
        outputs=[left_time, left_icon_display, left_image, right_time, right_icon_display, right_image],
    )
    prompt_box.submit(
        on_submit_generate,
        inputs=prompt_box,
        outputs=[left_time, left_icon_display, left_image, right_time, right_icon_display, right_image],
    )
    sample_prompt_btn.click(fn=on_click_sample_prompt, inputs=None, outputs=prompt_box)

demo.launch(share=True) 