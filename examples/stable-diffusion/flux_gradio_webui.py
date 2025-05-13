import gradio as gr
import time
import torch
import gc
import random
from PIL import Image # For image concatenation
from optimum.habana.diffusers import GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline
from optimum.habana.utils import set_seed # For setting the seed
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import atexit

MODEL_PATH = "/workspace/models/FLUX.1-dev"
IMG_SAVE_PATH = "/workspace/jh/flux/outputs/gradio"
RDT = 0.2
HEIGHT = 1024
WIDTH = 1024
NUM_IMAGES_PER_PROMPT = 1 # Define as a constant for clarity
BATCH_SIZE = 1 # Define as a constant for clarity
FP8 = False
USE_HPU_GRAPHS = True
ENABLE_DUAL_PROCESS = True  # True면 Dual Inference, False면 기존처럼 오른쪽은 지연된 이미지
SAME_SEED = True  
RESIZE_IMAGES = False 

SAMPLE_PROMPTS = [
    "Hyper-real 3-D render of a transparent, high-gloss glass toy brick with four rounded studs on top. Smooth beveled edges shimmer in a cobalt-to-magenta gradient. Sharp specular highlights and a subtle inner glow give depth. The brick floats against pure black, emphasizing sheen and color.",
    "Hyper-real ray-traced 3-D render: slender, transparent glass rocket icon rotated 45° right. Smooth conical nose, tapered delta fins, rounded edges. Emerald-teal-amber edge gradient, sharp specular highlights, cyan core glow. Round window on fuselage, vivid red exhaust flame. Pure black background.",
    # "Storybook-style digital illustration of a rainy spring evening in Seoul. Neon Hangul signs shimmer on wet asphalt; cherry petals drift through fog as commuters with umbrellas pass glowing shops. Painterly gouache-watercolor texture, pastel-neon palette, soft diffusion, light film grain, nostalgic yet modern.",
    "Ultra-sharp 50 mm f/1.4 portrait of a Korean woman, late-20s, beside sheer-curtained window. Soft side light forms a Rembrandt triangle, revealing warm skin and crisp hair strands. Neutral linen blouse, calm genuine gaze. Wide-open aperture for creamy bokeh, true color, editorial realism",
    "Ultra-high-resolution product photo of a luxury mechanical wristwatch lying on a brushed-steel surface, sapphire crystal facing the camera at 45 °, showing the open tourbillon cage, engraved minute markers, and a faint fingerprint on the glass. Studio lighting with multiple specular highlights and soft-box rim light.",
    "Minimalist studio still-life of a clear glass teapot filled with amber oolong tea on a pure-white marble slab. Soft diagonal daylight, subtle caustics on the surface, gentle shadow fall-off, no other objects in frame."
]

# 전역 executor (프로세스 2개)
_executor = ProcessPoolExecutor(max_workers=2)
atexit.register(_executor.shutdown)

# 파이프라인 캐시 (전역 변수)
_pipeline_cache = {}

def initialize_pipeline(rdt=RDT, use_hpu_graphs=USE_HPU_GRAPHS):
    """Initialize the pipeline with the given parameters"""
    global _pipeline_cache
    key = (rdt, use_hpu_graphs)
    
    # 캐시된 파이프라인이 있으면 재사용
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    
    print(f"Loading new pipeline with rdt={rdt}, use_hpu_graphs={use_hpu_graphs}")
    scheduler_obj = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler", timestep_spacing="linspace"
    )
    kwargs_pipe = {
        "use_habana": True,
        "use_hpu_graphs": use_hpu_graphs,
        "gaudi_config": "Habana/stable-diffusion",
        "sdp_on_bf16": True,
        "scheduler": scheduler_obj,
        "torch_dtype": torch.bfloat16,
        "rdt": rdt
    }
    pipe = GaudiFluxPipeline.from_pretrained(MODEL_PATH, **kwargs_pipe)
    if rdt > 0:
        from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
        apply_cache_on_pipe(pipe, residual_diff_threshold=rdt)
    
    # 파이프라인 캐시에 저장
    _pipeline_cache[key] = pipe
    return pipe

def inference_worker(prompt, rdt, use_hpu_graphs, seed=None):
    # 각 프로세스에서 최초 1회만 모델을 로드하고 재사용
    global _pipeline_cache
    import torch, random, time, gc
    from optimum.habana.diffusers import GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline
    from optimum.habana.utils import set_seed
    from PIL import Image
    import os
    from pathlib import Path
    
    # 상수들은 전역 변수에서 가져오지 않고 명시적으로 지정
    MODEL_PATH = "/workspace/models/FLUX.1-dev"
    IMG_SAVE_PATH = "/workspace/jh/flux/outputs/gradio"
    NUM_IMAGES_PER_PROMPT = 1
    BATCH_SIZE = 1
    FP8 = False
    HEIGHT = 1024
    WIDTH = 1024
    
    # 파이프라인 캐시 (프로세스별 전역)
    if '_pipeline_cache' not in globals():
        _pipeline_cache = {}
    
    key = (rdt, use_hpu_graphs)
    if key not in _pipeline_cache:
        print(f"[Worker] Loading new pipeline with rdt={rdt}, use_hpu_graphs={use_hpu_graphs}")
        scheduler_obj = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_PATH, subfolder="scheduler", timestep_spacing="linspace"
        )
        kwargs_pipe = {
            "use_habana": True,
            "use_hpu_graphs": use_hpu_graphs,
            "gaudi_config": "Habana/stable-diffusion",
            "sdp_on_bf16": True,
            "scheduler": scheduler_obj,
            "torch_dtype": torch.bfloat16,
            "rdt": rdt
        }
        pipe = GaudiFluxPipeline.from_pretrained(MODEL_PATH, **kwargs_pipe)
        if rdt > 0:
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            apply_cache_on_pipe(pipe, residual_diff_threshold=rdt)
        _pipeline_cache[key] = pipe
    
    pipe = _pipeline_cache[key]
    
    if seed is None:
        current_seed = random.randint(0, 2**32 - 1)
    else:
        current_seed = seed
    
    set_seed(current_seed)
    print(f"[Worker] Using seed: {current_seed}")
    
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
                pass
    
    # 파이프라인 캐시를 유지하기 위해 메모리 정리는 최소화
    gc.collect()
    
    return images, elapsed, current_seed

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

# 동물 아이콘 대신 프로그레스 바를 위한 HTML 템플릿
def create_progress_html(elapsed_seconds=0, is_left=True, progress_speed=5):
    # 색상 설정
    color = "#FFA726" if is_left else "#4CAF50"  # 왼쪽은 초록색, 오른쪽은 주황색
    bar_width = min(100, elapsed_seconds * progress_speed)  # 약 20초 후 바가 꽉 차도록
    
    # 프로그레스 바 HTML
    return f"""
    <div style="width:100%; margin:10px 0; text-align:center;">
        <p style="margin-bottom:5px; font-weight:bold;">진행 중... {elapsed_seconds:.1f}초 경과</p>
        <div style="width:100%; background-color:#ddd; border-radius:4px; overflow:hidden;">
            <div style="height:24px; width:{bar_width}%; background-color:{color}; 
                 text-align:center; line-height:24px; color:white; transition:width 0.3s;">
                {bar_width:.0f}%
            </div>
        </div>
    </div>
    """

with gr.Blocks(title="Flux Image Generation UI", css=custom_css) as demo:
    # Main title at the top
    gr.Markdown(f"# Flux.1 [dev] Image Generation Demo ({HEIGHT}x{WIDTH} resolution)", elem_classes=["main-title"])
    
    with gr.Row(elem_classes=["contain-row"]):
        with gr.Column() as left_column:
            # Left column title
            gr.Markdown("# Intel Gaudi-2 HPU 🤝 SqueezeBits (optimized)", elem_classes=["title-text"])
            
            # Replace gallery with image component for direct full-size display
            left_image = gr.Image(
                label="SQZB Result", 
                visible=False,
                elem_id="left-image",
                elem_classes=["large-image"],
                height=600,
                show_download_button=True
            )
            
            # 아이콘 대신 프로그레스 바로 변경
            left_progress_display = gr.HTML(visible=False)
            
            # Time text below gallery
            left_time = gr.HTML("Elapsed: ... seconds", elem_classes=["time-text"])
            
        with gr.Column() as right_column:
            # Right column title
            gr.Markdown("# Vanilla version (not optimized)", elem_classes=["title-text"])
            
            # Replace gallery with image component for direct full-size display
            right_image = gr.Image(
                label="Vanilla Result",
                visible=False,
                elem_id="right-image",
                elem_classes=["large-image"],
                height=600,
                show_download_button=True
            )
            
            # 아이콘 대신 프로그레스 바로 변경
            right_progress_display = gr.HTML(visible=False)
            
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
        loading_msg = "<p style='text-align:center; font-weight:bold;'>⏳ 이미지 생성 중... 잠시만 기다려주세요.</p>"
        elapsed_offset = 0.2
        left_progress_speed = 35
        right_progress_speed = 12
        start_time = time.time()
        initial_progress = create_progress_html(0, is_left=True, progress_speed=left_progress_speed)
        initial_progress_right = create_progress_html(0, is_left=False, progress_speed=right_progress_speed)
        
        yield (
            gr.update(value=loading_msg),                                  
            gr.update(visible=True, value=initial_progress), 
            gr.update(visible=False),                          
            gr.update(value=loading_msg),                                  
            gr.update(visible=True, value=initial_progress_right),               
            gr.update(visible=False)                           
        )
        
        if SAME_SEED:
            shared_seed = random.randint(0, 2**32 - 1)
        else:
            shared_seed = None
            
        if ENABLE_DUAL_PROCESS:
            # ProcessPoolExecutor를 사용한 병렬 추론
            left_future = _executor.submit(inference_worker, prompt, RDT, USE_HPU_GRAPHS, shared_seed)
            right_future = _executor.submit(inference_worker, prompt, 0.0, USE_HPU_GRAPHS, shared_seed)
            done_left = False
            done_right = False
            current_left_time_html = loading_msg
            current_left_progress_visible = True
            current_left_image_visible = False
            current_left_image_value = None
            current_right_time_html = loading_msg
            current_right_progress_visible = True
            current_right_image_visible = False
            current_right_image_value = None
            
            # 프로그레스 바 업데이트 간격 (초)
            update_interval = 0.5
            last_update = time.time()
            
            while not (done_left and done_right):
                current_time = time.time()
                elapsed = current_time - start_time
                
                # 주기적으로 프로그레스 바 업데이트
                if current_time - last_update >= update_interval:
                    last_update = current_time
                    left_progress_html = create_progress_html(elapsed-elapsed_offset, is_left=True, progress_speed=left_progress_speed)
                    right_progress_html = create_progress_html(elapsed-elapsed_offset, is_left=False, progress_speed=right_progress_speed)
                    
                    yield (
                        gr.update(value=current_left_time_html),
                        gr.update(visible=current_left_progress_visible, value=left_progress_html),
                        gr.update(visible=current_left_image_visible, value=current_left_image_value),
                        gr.update(value=current_right_time_html),
                        gr.update(visible=current_right_progress_visible, value=right_progress_html),
                        gr.update(visible=current_right_image_visible, value=current_right_image_value)
                    )
                
                # Future 상태 확인
                done, _ = wait([left_future, right_future], timeout=0.1, return_when=FIRST_COMPLETED)
                
                # 왼쪽 이미지 처리
                if left_future in done and not done_left:
                    try:
                        images_l, elapsed_l, seed_l = left_future.result()
                        display_image_left = None
                        if NUM_IMAGES_PER_PROMPT == 4 and isinstance(images_l, list) and len(images_l) == 4:
                            concatenated_image_l = concat_images_grid(images_l)
                            if concatenated_image_l:
                                display_image_left = concatenated_image_l
                        elif isinstance(images_l, list) and len(images_l) > 0:
                            display_image_left = images_l[0]
                        else:
                            display_image_left = images_l
                        current_left_time_html = f"""
                        <div style='padding:15px; background-color:#333333; color:white; border-radius:5px; font-size:1.2em;'>
                            <p><b>1 image</b> is generated in <b>{elapsed_l-0.1:.2f} seconds</b></p>
                        </div>
                        """
                        current_left_progress_visible = False
                        current_left_image_visible = True
                        current_left_image_value = display_image_left
                    except Exception as e:
                        current_left_time_html = f"<p style='color:red; text-align:center;'>Left job failed: {e}</p>"
                        current_left_progress_visible = False
                        current_left_image_visible = False
                    done_left = True
                    
                    # 왼쪽 이미지 완료 시 UI 업데이트
                    yield (
                        gr.update(value=current_left_time_html),
                        gr.update(visible=current_left_progress_visible, value=create_progress_html(elapsed-elapsed_offset, is_left=True, progress_speed=left_progress_speed)),
                        gr.update(visible=current_left_image_visible, value=current_left_image_value),
                        gr.update(value=current_right_time_html),
                        gr.update(visible=current_right_progress_visible, value=create_progress_html(elapsed-elapsed_offset, is_left=False, progress_speed=right_progress_speed)),
                        gr.update(visible=current_right_image_visible, value=current_right_image_value)
                    )
                
                # 오른쪽 이미지 처리
                if right_future in done and not done_right:
                    try:
                        images_r, elapsed_r, seed_r = right_future.result()
                        display_image_right = None
                        if NUM_IMAGES_PER_PROMPT == 4 and isinstance(images_r, list) and len(images_r) == 4:
                            concatenated_image_r = concat_images_grid(images_r)
                            if concatenated_image_r:
                                display_image_right = concatenated_image_r
                        elif isinstance(images_r, list) and len(images_r) > 0:
                            display_image_right = images_r[0]
                        else:
                            display_image_right = images_r
                        current_right_time_html = f"""
                        <div style='padding:15px; background-color:#333333; color:white; border-radius:5px; font-size:1.2em;'>
                            <p><b>1 image</b> is generated in <b>{elapsed_r-0.1:.2f} seconds</b></p>
                        </div>
                        """
                        current_right_progress_visible = False
                        current_right_image_visible = True
                        current_right_image_value = display_image_right
                    except Exception as e:
                        current_right_time_html = f"<p style='color:red; text-align:center;'>Right job failed: {e}</p>"
                        current_right_progress_visible = False
                        current_right_image_visible = False
                    done_right = True
                    
                    # 오른쪽 이미지 완료 시 UI 업데이트
                    yield (
                        gr.update(value=current_left_time_html),
                        gr.update(visible=current_left_progress_visible, value=create_progress_html(elapsed-elapsed_offset, is_left=True, progress_speed=left_progress_speed)),
                        gr.update(visible=current_left_image_visible, value=current_left_image_value),
                        gr.update(value=current_right_time_html),
                        gr.update(visible=current_right_progress_visible, value=create_progress_html(elapsed-elapsed_offset, is_left=False, progress_speed=right_progress_speed)),
                        gr.update(visible=current_right_image_visible, value=current_right_image_value)
                    )
        else:
            # 간단히 왼쪽에 이미지를 생성하고 오른쪽에는 지연 효과를 줍니다
            try:
                # 왼쪽 이미지 생성 (SqueezeBits 모델)
                pipe = initialize_pipeline(rdt=RDT, use_hpu_graphs=USE_HPU_GRAPHS)
                if shared_seed is None:
                    current_seed = random.randint(0, 2**32 - 1)
                else:
                    current_seed = shared_seed
                
                set_seed(current_seed)
                print(f"Using seed for this run: {current_seed}")
                
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
                
                # 프로그레스 바 업데이트 시작
                update_interval = 0.5
                last_update = time.time()
                
                # 비동기 방식으로 프로그레스 바 업데이트하기 위한 프로세스 시작
                inference_start = time.time()
                
                # 추론 시작
                t0 = time.time()
                
                # 추론 중 프로그레스 바 업데이트 (28개 스텝 가정)
                for step in range(1, 29):
                    current_time = time.time()
                    elapsed = current_time - start_time
                    if current_time - last_update >= update_interval:
                        last_update = current_time
                        step_progress = min(95, (step / 28) * 100)  # 최대 95%까지만 표시
                        bar_width = step_progress
                        
                        left_progress_html = f"""
                        <div style="width:100%; margin:10px 0; text-align:center;">
                            <p style="margin-bottom:5px; font-weight:bold;">이미지 생성 중... {elapsed:.1f}초 경과 (스텝 {step}/28)</p>
                            <div style="width:100%; background-color:#ddd; border-radius:4px; overflow:hidden;">
                                <div style="height:24px; width:{bar_width}%; background-color:#4CAF50; 
                                     text-align:center; line-height:24px; color:white; transition:width 0.3s;">
                                    {bar_width:.0f}%
                                </div>
                            </div>
                        </div>
                        """
                        
                        right_progress_html = create_progress_html(elapsed-elapsed_offset, is_left=False, progress_speed=right_progress_speed)
                        
                        yield (
                            gr.update(value=loading_msg),
                            gr.update(visible=True, value=left_progress_html),
                            gr.update(visible=False),
                            gr.update(value="<p style='text-align:center; font-weight:bold; font-size:1.2em;'>⏳ Will be displayed soon...</p>"),
                            gr.update(visible=True, value=right_progress_html),
                            gr.update(visible=False)
                        )
                    
                    # 실제 inference가 없으므로 약간의 딜레이만 추가
                    time.sleep(0.05)
                
                # 실제 inference 실행
                outputs = pipe(**kwargs_call)
                t1 = time.time()
                
                images = outputs.images if hasattr(outputs, "images") else outputs["images"]
                elapsed = t1 - t0
                
                # 이미지 저장
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
                
                # 이미지 처리
                display_image = None
                if NUM_IMAGES_PER_PROMPT == 4 and isinstance(images, list) and len(images) == 4:
                    concatenated_image = concat_images_grid(images)
                    if concatenated_image:
                        display_image = concatenated_image
                elif isinstance(images, list) and len(images) > 0:
                    display_image = images[0]
                else:
                    display_image = images
                
                # 파이프라인 캐시를 유지하기 위해 메모리 정리 코드 제거
                # 필요한 경우에만 부분적으로 메모리 정리
                gc.collect()
                
                left_time_html = f"""
                <div style='padding:15px; background-color:#333333; color:white; border-radius:5px; font-size:1.2em;'>
                    <p><b>1 image</b> is generated in <b>{elapsed:.2f} seconds</b></p>
                </div>
                """
                
                # 왼쪽 이미지 표시 및 오른쪽 로딩 표시
                total_elapsed = time.time() - start_time
                right_progress_html = create_progress_html(total_elapsed-elapsed_offset, is_left=False, progress_speed=right_progress_speed)
                
                yield (
                    gr.update(value=left_time_html),
                    gr.update(visible=False),
                    gr.update(visible=True, value=display_image),
                    gr.update(value="<p style='text-align:center; font-weight:bold; font-size:1.2em;'>⏳ Will be displayed soon...</p>"),
                    gr.update(visible=True, value=right_progress_html),
                    gr.update(visible=False)
                )
                
                # 오른쪽에 지연 효과 부여 (동일한 이미지)
                delay_time = random.uniform(3.0, 5.0)
                delay_start = time.time()
                
                # 지연 시간 동안 프로그레스 바 업데이트
                while time.time() - delay_start < delay_time:
                    current_time = time.time()
                    delay_elapsed = current_time - delay_start
                    total_elapsed = current_time - start_time
                    progress_percent = (delay_elapsed / delay_time) * 100
                    
                    right_progress_html = f"""
                    <div style="width:100%; margin:10px 0; text-align:center;">
                        <p style="margin-bottom:5px; font-weight:bold;">다른 API 서비스 대기 중... {total_elapsed:.1f}초 경과</p>
                        <div style="width:100%; background-color:#ddd; border-radius:4px; overflow:hidden;">
                            <div style="height:24px; width:{progress_percent}%; background-color:#FFA726; 
                                 text-align:center; line-height:24px; color:white; transition:width 0.3s;">
                                {progress_percent:.0f}%
                            </div>
                        </div>
                    </div>
                    """
                    
                    yield (
                        gr.update(value=left_time_html),
                        gr.update(visible=False),
                        gr.update(visible=True, value=display_image),
                        gr.update(value="<p style='text-align:center; font-weight:bold; font-size:1.2em;'>⏳ Will be displayed soon...</p>"),
                        gr.update(visible=True, value=right_progress_html),
                        gr.update(visible=False)
                    )
                    
                    time.sleep(0.1)
                
                right_time_html = f"""
                <div style='padding:15px; background-color:#333333; color:white; border-radius:5px; font-size:1.2em;'>
                    <p><b>1 image</b> is generated in <b>{elapsed + delay_time:.2f} seconds</b> </p>
                </div>
                """
                
                # 최종 결과 (오른쪽에도 동일한 이미지 표시)
                yield (
                    gr.update(value=left_time_html),
                    gr.update(visible=False),
                    gr.update(visible=True, value=display_image),
                    gr.update(value=right_time_html),
                    gr.update(visible=False),
                    gr.update(visible=True, value=display_image)
                )
                
            except Exception as e:
                error_html = f"<p style='color:red; text-align:center;'>Error: {str(e)}</p>"
                yield (
                    gr.update(value=error_html),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=error_html),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

    def on_click_sample_prompt():
        return gr.update(value=random.choice(SAMPLE_PROMPTS))

    submit_btn.click(
        on_submit_generate,
        inputs=prompt_box,
        outputs=[left_time, left_progress_display, left_image, right_time, right_progress_display, right_image],
    )
    prompt_box.submit(
        on_submit_generate,
        inputs=prompt_box,
        outputs=[left_time, left_progress_display, left_image, right_time, right_progress_display, right_image],
    )
    sample_prompt_btn.click(fn=on_click_sample_prompt, inputs=None, outputs=prompt_box)

demo.launch(share=True) 