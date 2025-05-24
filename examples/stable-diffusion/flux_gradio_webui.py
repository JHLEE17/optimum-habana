import gradio as gr
import time
import torch
import gc
import random
import argparse
import asyncio
import os
import base64
import requests
import replicate
import json
import traceback
import sys
from PIL import Image # For image concatenation
from optimum.habana.diffusers import GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline
from optimum.habana.utils import set_seed # For setting the seed
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import atexit
# from dotenv import load_dotenv
import io

# Load environment variables from .env file
# load_dotenv()

FP8 = True
RDT = 0.12
MODEL_PATH = "/workspace/models/FLUX.1-dev"
IMG_SAVE_PATH = "/workspace/jh/flux/outputs/gradio"
HEIGHT = 1024
WIDTH = 1024
NUM_IMAGES_PER_PROMPT = 1 # Define as a constant for clarity
BATCH_SIZE = 1 # Define as a constant for clarity
USE_HPU_GRAPHS = True
ENABLE_DUAL_PROCESS = True  # True면 Dual Inference, False면 기존처럼 오른쪽은 지연된 이미지
SAME_SEED = True  
RESIZE_IMAGES = False 
GUIDANCE_SCALE = 3.5

# NOTE: Set this to the actual path of your FP8 quantization configuration JSON file.
QUANT_CONFIG_FILE_PATH = "quantization/flux/quantize_config.json" 


# Parse command line arguments
parser = argparse.ArgumentParser(description="Flux Image Generation WebUI")
parser.add_argument("--compare", action="store_true", help="Enable comparison mode with multiple APIs")
parser.add_argument("--fal-key", type=str, help="Fal.ai API key")
parser.add_argument("--replicate-key", type=str, help="Replicate API key")
parser.add_argument("--wavespeed-key", type=str, help="WaveSpeed API key")
parser.add_argument("--runware-key", type=str, help="Runware API key")
parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
parser.add_argument("--skip-local", action="store_true", help="Skip local model inference (useful for testing the UI)")
parser.add_argument("--log-file", type=str, default="api_debug.log", help="File to save API debug logs")
parser.add_argument("--save-responses", action="store_true", help="Save API responses to files")
args = parser.parse_args()

# Set debug mode
DEBUG_MODE = args.debug
if DEBUG_MODE:
    print("Debug mode enabled - detailed logging will be shown")

# Skip local models flag
SKIP_LOCAL_MODELS = args.skip_local
if SKIP_LOCAL_MODELS:
    print("Local model inference will be skipped (for UI testing only)")

# API keys for external services (when in compare mode)
# .env 로부터 API 키 로드 (있는 경우) 또는 명령줄 인자에서 로드
FAL_KEY = args.fal_key or os.getenv("FAL_KEY")
REPLICATE_API_TOKEN = args.replicate_key or os.getenv("REPLICATE_API_TOKEN")
WAVESPEED_API_KEY = args.wavespeed_key or os.getenv("WAVESPEED_API_KEY") 
RUNWARE_API_KEY = args.runware_key or os.getenv("RUNWARE_API_KEY")

# For debugging
print(f"API Keys available: FAL_KEY={bool(FAL_KEY)}, REPLICATE={bool(REPLICATE_API_TOKEN)}, WAVESPEED={bool(WAVESPEED_API_KEY)}, RUNWARE={bool(RUNWARE_API_KEY)}")

# API service model definitions
FAL_MODEL = "fal-ai/flux/dev"
REPLICATE_MODEL = "black-forest-labs/flux-dev"
WAVESPEED_MODELS = {
    "wavespeed_dev": "wavespeed-ai/flux-dev",
    "wavespeed_fast": "wavespeed-ai/flux-dev-ultra-fast",
}
RUNWARE_MODEL = "runware:101@1"

# Initialize API clients
if args.compare:
    # Only initialize these if --compare flag is used
    try:
        if REPLICATE_API_TOKEN:
            replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            print("Replicate client initialized")
        else:
            print("Warning: REPLICATE_API_TOKEN not found, skipping initialization")
        # Other clients will be initialized as needed
    except Exception as e:
        print(f"Warning: Failed to initialize some API clients: {e}")
        print("Comparison mode may not work properly without all API keys")

SAMPLE_PROMPTS = [
    "Hyper-real 3-D render of a transparent, high-gloss glass toy brick with four rounded studs on top. Smooth beveled edges shimmer in a cobalt-to-magenta gradient. Sharp specular highlights and a subtle inner glow give depth. The brick floats against pure black, emphasizing sheen and color.",
    "Hyper-real ray-traced 3-D render: slender, transparent glass rocket icon rotated 45° right. Smooth conical nose, tapered delta fins, rounded edges. Emerald-teal-amber edge gradient, sharp specular highlights, cyan core glow. Round window on fuselage, vivid red exhaust flame. Pure black background.",
    # "Storybook-style digital illustration of a rainy spring evening in Seoul. Neon Hangul signs shimmer on wet asphalt; cherry petals drift through fog as commuters with umbrellas pass glowing shops. Painterly gouache-watercolor texture, pastel-neon palette, soft diffusion, light film grain, nostalgic yet modern.",
    "Ultra-sharp 50 mm f/1.4 portrait of a Korean woman, late-20s, beside sheer-curtained window. Soft side light forms a Rembrandt triangle, revealing warm skin and crisp hair strands. Neutral linen blouse, calm genuine gaze. Wide-open aperture for creamy bokeh, true color, editorial realism",
    "Ultra-sharp 50 mm f/1.4 portrait of a Korean woman, late-20s, beside sheer-curtained window. Soft side light forms a Rembrandt triangle, revealing warm skin and crisp hair strands. Neutral linen blouse, calm genuine gaze. Wide-open aperture for creamy bokeh, true color, editorial realism",
    # "Ultra-high-resolution product photo of a luxury mechanical wristwatch lying on a brushed-steel surface, sapphire crystal facing the camera at 45 °, showing the open tourbillon cage, engraved minute markers, and a faint fingerprint on the glass. Studio lighting with multiple specular highlights and soft-box rim light.",
    "Minimalist studio still-life of a clear glass teapot filled with amber oolong tea on a pure-white marble slab. Soft diagonal daylight, subtle caustics on the surface, gentle shadow fall-off, no other objects in frame.",
    "Classic American traditional tattoo style Winston Churchill looking at iPhone.. Bold lines, nautical elements, retro flair, symbolic",
    "A serene portrait of Yuzuru Otonashi from *Angel Beats!* in Studio Ghibli's soft, painterly style. He stands in a sunlit field of wildflowers, wearing his crisp white school uniform with a gentle breeze rustling his dark, slightly messy hair. His warm brown eyes reflect quiet determination and kindness, with soft Ghibli-style shading enhancing his youthful features. Delicate cherry blossom petals drift in the air around him, glowing under golden-hour sunlight. The background blends lush greenery and distant rolling hills, evoking Ghibli's dreamy landscapes. Subtle ethereal glow and muted pastel tones create a nostalgic, melancholic yet hopeful atmosphere, capturing the emotional depth of the series while maintaining Ghibli's whimsical charm.",
    "A breathtaking panorama of the Lake District at dawn, where gentle hills roll into the distance, their slopes adorned with vibrant patches of heather and lush green grass. A serene lake mirrors the soft pastels of the early morning sky, reflecting hues of lavender and peach as sunlight begins to break through the mist. Wisps of fog linger over the water, creating an ethereal atmosphere. Majestic, rugged peaks rise in the background, their rocky faces dusted with the remnants of overnight rain, glistening under the soft golden light. The scene is tranquil yet invigorating, evoking a sense of peace and wonder. Capture this landscape in a painterly style, emphasizing the interplay of light and shadow, with a focus on texture and depth, reminiscent of the works of J.M.W. Turner.",
    "A charismatic speaker is captured mid-speech, his [short, tousled brown] hairs lightly messy on top. He has a round face, clean-shaven, and wears [rounded rectangular glasses with dark rims]. He is holding a black microphone in his right hand, speaking passionately. His expression is animated as he gestures with his left hand. Dressed in [a light blue sweater over a white t-shirt]. The background is blurred, showcasing a white banner with logos, suggesting a professional [conference] setting.",
    "a silhouette of a lone surfer riding a massive wave under a sunset sky, water",
    "a futuristic motorcycle speeding along a neon-lit highway at night, streaks of light trailing behind",
    "A rugged old man in a heavy wool coat stands against a raging sea, his weathered hands gripping the rail of a storm-battered lighthouse. Waves crash violently against the rocky shore, and the sky is painted in deep purples and oranges as the sun sets behind rolling clouds. His eyes tell a story of solitude, resilience, and untold tales",
]

# 전역 executor (프로세스 2개)
_executor = ProcessPoolExecutor(max_workers=2)
atexit.register(_executor.shutdown)

# 파이프라인 캐시 (전역 변수)
_pipeline_cache = {}

# Create debug log file
log_file_path = args.log_file
with open(log_file_path, "w") as f:
    f.write(f"===== API Debug Log Started at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    
def log_debug(message):
    """Log a debug message to console and file"""
    timestamp = time.strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(log_file_path, "a") as f:
            f.write(f"{full_message}\n")
            f.flush()  # Force writing to disk
    except Exception as e:
        print(f"Error writing to log file: {e}")

def save_response(name, data):
    """Save API response data to a file"""
    if not args.save_responses:
        return
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    try:
        with open(filename, "w") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2)
            else:
                f.write(str(data))
        log_debug(f"Saved {name} response to {filename}")
    except Exception as e:
        log_debug(f"Error saving response: {e}")

# API service functions for comparison mode
async def call_runware_api_async(prompt, height=HEIGHT, width=WIDTH, seed=None):
    """Call the Runware API for image generation (async version)"""
    try:
        import sys
        sys.path.append(".")  # Make sure runware module can be imported
        from runware import Runware, IImageInference
        
        t0 = time.time()
        
        # Initialize Runware client
        runware = Runware(api_key=RUNWARE_API_KEY)
        await runware.connect()
        
        # Prepare request
        request_image = IImageInference(
            positivePrompt=prompt,
            model=RUNWARE_MODEL,
            numberResults=1,
            height=height,
            width=width,
            steps=28,
            CFGScale=GUIDANCE_SCALE,
            outputType="URL",
            outputFormat="png",
            seed=seed,
        )
        
        # Get images
        images = await runware.imageInference(requestImage=request_image)
        
        t1 = time.time()
        elapsed = t1 - t0
        
        if not images:
            return None, elapsed
            
        # Download the image from URL
        image_url = images[0].imageURL
        response = requests.get(image_url, timeout=60)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        
        # Save image if needed
        if IMG_SAVE_PATH:
            timestamp = int(time.time())
            save_path = os.path.join(IMG_SAVE_PATH, f"runware_{timestamp}.png")
            img.save(save_path)
        
        return img, elapsed
    except Exception as e:
        print(f"Error calling Runware API: {e}")
        return None, 0

def call_runware_api(prompt, height=HEIGHT, width=WIDTH, seed=None):
    """Non-async wrapper for Runware API"""
    try:
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(call_runware_api_async(prompt, height, width, seed))
        loop.close()
        return result
    except Exception as e:
        print(f"Error in Runware API wrapper: {e}")
        return None, 0

def call_fal_api(prompt, height=HEIGHT, width=WIDTH, timeout=30, seed=None):
    """Call the fal.ai API for image generation"""
    try:
        # Try to import fal_client only when needed
        try:
            import fal_client
            log_debug("fal_client module imported successfully")
        except ImportError:
            log_debug("fal_client module not found. Please install with 'pip install fal-client'")
            return None, 0, None
        
        t0 = time.time()
        
        # Ensure the API key is set
        if FAL_KEY:
            os.environ["FAL_KEY"] = FAL_KEY
            log_debug(f"Set FAL_KEY in environment (length: {len(FAL_KEY)})")
        else:
            log_debug("No FAL_KEY available in environment")
            return None, 0, "No API key available"
        
        log_debug(f"Calling fal.ai API with prompt: '{prompt[:30]}...' and timeout: {timeout}s")
        
        # Call fal.ai API with timeout
        result = fal_client.run(
            FAL_MODEL,
            arguments={
                "prompt": prompt,
                "image_size": {
                    "width": width,
                    "height": height,
                }, #"square_hd" if height == width else "portrait_hd" if height > width else "landscape_hd",
                "seed": seed,
                "num_images": 1,
                "num_inference_steps": 28,
                "guidance_scale": GUIDANCE_SCALE,
            },
            timeout=timeout,
        )
        
        log_debug(f"fal.ai API response received with keys: {list(result.keys())}")
        
        # Download image from URL
        image_url = result["images"][0]["url"]
        log_debug(f"Image URL: {image_url[:50]}...")
        
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
        log_debug("Image downloaded successfully")
        
        img = Image.open(io.BytesIO(response.content))
        log_debug(f"Image opened successfully: {img.size}")
        
        t1 = time.time()
        elapsed = t1 - t0
        log_debug(f"Total fal.ai process completed in {elapsed:.2f}s")
        
        # Save image if needed
        if IMG_SAVE_PATH:
            timestamp = int(time.time())
            save_path = os.path.join(IMG_SAVE_PATH, f"fal_{timestamp}.png")
            img.save(save_path)
            log_debug(f"Image saved to {save_path}")
        
        # Save response for debugging
        save_response("fal", result)
        
        return img, elapsed, None
    except Exception as e:
        error_message = f"Error in call_fal_api: {str(e)}"
        log_debug(error_message)
        log_debug(f"Traceback: {traceback.format_exc()}")
        return None, 0, error_message

def call_replicate_api(prompt, height=HEIGHT, width=WIDTH, timeout=60, seed=None):
    """Call the Replicate API for image generation"""
    try:
        t0 = time.time()
        
        # Call Replicate API with timeout
        result = replicate_client.run(
            REPLICATE_MODEL,
            input={
                "prompt": prompt,
                "height": height,
                "width": width,
                "guidance": GUIDANCE_SCALE,
                "seed": seed,
                "num_inference_steps": 28,
                "num_outputs": 1,
            },
            timeout=timeout,
        )
        
        image_url = result[0] if isinstance(result, list) else result
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        
        t1 = time.time()
        elapsed = t1 - t0
        
        # Save image if needed
        if IMG_SAVE_PATH:
            timestamp = int(time.time())
            save_path = os.path.join(IMG_SAVE_PATH, f"replicate_{timestamp}.png")
            img.save(save_path)
        
        return img, elapsed, image_url
    except Exception as e:
        print(f"Error calling Replicate API: {e}")
        return None, 0, None

def call_wavespeed_api(prompt, model_type="fast", height=HEIGHT, width=WIDTH, timeout=60, seed=None):
    """Call the WaveSpeed API for image generation"""
    try:
        t0 = time.time()
        
        # Determine which WaveSpeed model to use
        if model_type == "fast":
            model_slug = WAVESPEED_MODELS["wavespeed_fast"]
        else:
            model_slug = WAVESPEED_MODELS["wavespeed_dev"]
        
        # Prepare API request
        submit_url = f"https://api.wavespeed.ai/api/v2/{model_slug}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {WAVESPEED_API_KEY}",
        }
        payload = {
            "prompt": prompt,
            "size": f"{width}*{height}",
            "num_images": 1,
            "enable_base64_output": True,
            "enable_safety_checker": True,
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": 28,
            "seed": seed,
            "strength": 0.8,
        }
        
        # Submit job
        resp = requests.post(submit_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        task_id = resp.json()["data"]["id"]
        
        # Poll for results with timeout
        result_url = f"https://api.wavespeed.ai/api/v2/predictions/{task_id}/result"
        max_poll_time = t0 + timeout
        
        while time.time() < max_poll_time:
            res = requests.get(result_url, headers=headers, timeout=30)
            res.raise_for_status()
            data = res.json()["data"]
            status = data["status"]
            
            if status == "completed":
                output = data["outputs"][0]
                # Handle URL or base64
                if output.startswith("http"):
                    img_resp = requests.get(output, timeout=30)
                    img_resp.raise_for_status()
                    content = img_resp.content
                    ext = "png"
                elif output.startswith("data:image"):
                    header, b64 = output.split(",", 1)
                    ext = header.split("/")[1].split(";")[0]  # png / webp ...
                    content = base64.b64decode(b64)
                else:
                    content = base64.b64decode(output)
                    ext = "png"
                
                # Create image from bytes
                img = Image.open(io.BytesIO(content))
                
                t1 = time.time()
                elapsed = t1 - t0
                
                # Save image if needed
                if IMG_SAVE_PATH:
                    timestamp = int(time.time())
                    save_path = os.path.join(IMG_SAVE_PATH, f"wavespeed_{model_type}_{timestamp}.{ext}")
                    img.save(save_path)
                
                return img, elapsed
            
            if status == "failed":
                raise RuntimeError(data.get("error", "WaveSpeed task failed"))
            
            time.sleep(1)  # wait then poll again
            
        # If we get here, we've timed out
        raise TimeoutError(f"WaveSpeed API polling timed out after {timeout} seconds")
        
    except Exception as e:
        print(f"Error calling WaveSpeed API ({model_type}): {e}")
        return None, 0

def initialize_pipeline(rdt=RDT, use_hpu_graphs=USE_HPU_GRAPHS):
    """Initialize the pipeline with the given parameters"""
    global _pipeline_cache
    key = (rdt, use_hpu_graphs, FP8)  # Add FP8 to the cache key
    
    # 캐시된 파이프라인이 있으면 재사용
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    
    print(f"Loading new pipeline with rdt={rdt}, use_hpu_graphs={use_hpu_graphs}, FP8={FP8}")
    scheduler_obj = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler", timestep_spacing="linspace"
    )
    
    # 1. Pipeline 초기화
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
    
    # 2. Quantization 수행 (FP8 옵션이 활성화된 경우)
    if FP8:
        print(f"Applying FP8 quantization")
        pipe.quantize(quant_mode="quantize", quant_config_path=QUANT_CONFIG_FILE_PATH)
    
    # 3. RDT 설정 (기존 코드와 동일)
    if rdt > 0:
        from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
        apply_cache_on_pipe(pipe, residual_diff_threshold=rdt)
    
    # 파이프라인 캐시에 저장
    _pipeline_cache[key] = pipe
    return pipe

def inference_worker(prompt, rdt, use_hpu_graphs, seed=None, force_no_fp8: bool = False):
    # 각 프로세스에서 최초 1회만 모델을 로드하고 재사용
    global _pipeline_cache
    import torch, random, time, gc
    from optimum.habana.diffusers import GaudiFlowMatchEulerDiscreteScheduler, GaudiFluxPipeline
    from optimum.habana.utils import set_seed
    from PIL import Image
    import os
    from pathlib import Path
    
    # 파이프라인 캐시 (프로세스별 전역)
    if '_pipeline_cache' not in globals():
        _pipeline_cache = {}
    
    # Determine actual quantization state for this worker based on global FP8 and force_no_fp8 flag
    apply_quantization_for_worker = FP8 and not force_no_fp8
    
    key = (rdt, use_hpu_graphs, apply_quantization_for_worker) # Use actual quantization state for cache key
    if key not in _pipeline_cache:
        print(f"[Worker] Loading new pipeline with rdt={rdt}, use_hpu_graphs={use_hpu_graphs}, FP8 (applied)={apply_quantization_for_worker}")
        
        # 1. 스케줄러 초기화
        scheduler_obj = GaudiFlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_PATH, subfolder="scheduler", timestep_spacing="linspace"
        )
        
        # 2. 파이프라인 초기화
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
        
        # 3. Quantization 적용 (필요한 경우)
        if apply_quantization_for_worker:
            print(f"[Worker] Applying FP8 quantization")
            pipe.quantize(quant_mode="quantize", quant_config_path=QUANT_CONFIG_FILE_PATH)
        
        # 4. RDT 설정 (기존 코드와 동일)
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
        "guidance_scale": GUIDANCE_SCALE,
    }
    
    # FP8 quantization이 이미 별도로 적용되었으므로 quant_mode 파라미터 제거
    
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
    margin-bottom: 0px !important;
    font-weight: bold !important;
    font-size: 1.3em !important;
    text-align: center !important;
    display: inline-block !important;
}

/* Time text styling */
.time-text {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
    margin-left: 10px !important;
    padding: 5px !important;
    background-color: rgba(0, 0, 0, 0.1) !important;
    border-radius: 5px !important;
    font-size: 1.0em !important;
    line-height: 1 !important;
    height: auto !important;
    min-height: 8px !important;
    display: inline-block !important;
    vertical-align: middle !important;
    text-align: center !important;
}

/* Full viewport height containers */
.contain-row {
    min-height: 420px !important;
}

/* Image styling - expand to fill container */
.large-image {
    width: 100% !important;
    height: 640px !important; /* Default height for non-compare mode. Increased from 300px */
    margin: 0 auto !important; /* Centers the .large-image block if its container is wider. */
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important; /* Good practice. */
}

/* CSS for the actual <img> tag within the .large-image container */
.large-image img {
    display: block !important;
    max-width: 100% !important;
    max-height: 100% !important;
    object-fit: contain !important;
    border: 2px solid #444 !important; /* Merged from original */
    border-radius: 8px !important; /* Merged from original */
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

/* Seed label styling to align with the input */
.seed-label {
    margin: auto 0 !important;
    padding-right: 5px !important;
    font-weight: normal !important;
    white-space: nowrap !important;
}

/* Make seed input shorter */
#seed-row input, #seed-row .form {
    height: 25px !important;
    min-height: 25px !important;
    line-height: 25px !important;
}

#seed-row .form {
    margin: 0 !important;
    padding: 0 !important;
}

#seed-row .block {
    padding: 0 !important;
}

/* Make checkbox container smaller */
#random-seed-checkbox label span:first-of-type {
    padding: 0 !important;
    margin-right: 5px !important;
}

#random-seed-checkbox {
    width: auto !important;
    min-width: 0 !important;
}

/* Image sizing for comparison layout */
.compact-column {
    padding: 5px !important;
    margin: 0 !important;
}

.compact-column .large-image {
    height: 450px !important;
    width: 450px !important;
}

.compact-column .time-text {
    min-height: 20px !important;
    font-size: 1.2em !important;
}

/* Title container to hold title and time together */
.title-container {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-bottom: 5px !important;
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

def on_submit_generate(prompt, user_seed=None, use_random_seed=True):
    # Determine seed for this generation
    if use_random_seed:
        if SAME_SEED:
            shared_seed = random.randint(0, 2**32 - 1)
        else:
            shared_seed = None
    else:
        try:
            # Convert to integer and ensure it's within allowed range
            shared_seed = int(user_seed)
            if shared_seed < 0 or shared_seed > 2**32 - 1:
                shared_seed = random.randint(0, 2**32 - 1)
                print(f"Seed out of range, using random seed instead: {shared_seed}")
        except (ValueError, TypeError):
            shared_seed = random.randint(0, 2**32 - 1)
            print(f"Invalid seed input, using random seed instead: {shared_seed}")
    
    print(f"on_submit_generate called with prompt: {prompt}, seed: {shared_seed}, random: {use_random_seed}, compare: {args.compare}")
    
    # Handle compare mode (2x4 grid)
    if args.compare:
        print("Running in compare mode")
        return on_submit_generate_compare(prompt, shared_seed)
    
    # Original mode (1x2 grid)
    loading_msg = "<p style='text-align:center; font-weight:bold;'>⏳ 이미지 생성 중... 잠시만 기다려주세요.</p>"
    elapsed_offset = 0.3
    left_progress_speed = 34
    right_progress_speed = 11
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
    
    if ENABLE_DUAL_PROCESS:
        # ProcessPoolExecutor를 사용한 병렬 추론
        # Left pipeline respects global FP8 setting
        left_future = _executor.submit(inference_worker, prompt, RDT, USE_HPU_GRAPHS, shared_seed, force_no_fp8=False)
        # Right pipeline explicitly disables FP8 quantization
        right_future = _executor.submit(inference_worker, prompt, 0.0, USE_HPU_GRAPHS, shared_seed, force_no_fp8=True)
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
                "guidance_scale": GUIDANCE_SCALE,
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

# New function to handle compare mode with 2x4 grid of images
def on_submit_generate_compare(prompt, shared_seed):
    """
    Handle generation in compare mode with multiple API calls in a 2x4 grid
    Row 1: [right_future, fal, replicate, runware]
    Row 2: [left_future, wavespeed_fast, wavespeed_dev, empty]
    """
    log_debug(f"Starting comparison mode generation with prompt: '{prompt[:50]}...'")
    log_debug(f"Using seed: {shared_seed}")
    
    # Define placeholders for all 8 positions with their corresponding indices
    services = {
        0: "vanilla",         # right_future
        1: "fal",             # fal.ai
        2: "replicate",       # replicate
        3: "runware",         # runware
        4: "sqzb",            # left_future (SQZB Optimized)
        5: "wavespeed_fast",  # wavespeed fast
        6: "wavespeed_dev",   # wavespeed dev
        7: "empty"            # empty cell
    }
    
    # Initialize all cells with default values
    cell_states = []
    for i in range(8):
        cell_states.append([
            "<p style='text-align:center;'>Ready</p>",  # time_html
            gr.HTML(visible=False),                     # progress_html
            None                                        # image
        ])
    
    # Send initial state - Generate exactly 24 outputs (8 cells × 3 components)
    log_debug("Initial UI update...")
    try:
        outputs = yield_ui_update(cell_states)
        if outputs:
            log_debug(f"Yielding {len(outputs)} outputs")
            yield outputs
            log_debug("Initial UI update successful")
        else:
            log_debug("Failed to generate initial UI outputs")
            return
    except Exception as e:
        log_debug(f"Error in initial UI update: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        return
    
    # 모든 셀 상태 업데이트 - 서비스별 상태 메시지 표시
    for i in range(8):
        if i == 0:  # Vanilla 모델
            if SKIP_LOCAL_MODELS:
                cell_states[i][0] = "<p style='color:gray; text-align:center;'>Skipped (--skip-local)</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 1:  # Fal.ai
            if not FAL_KEY:
                cell_states[i][0] = "<p style='color:orange; text-align:center;'>No fal.ai API key</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 2:  # Replicate
            if not REPLICATE_API_TOKEN:
                cell_states[i][0] = "<p style='color:orange; text-align:center;'>No Replicate API key</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 3:  # Runware
            if not RUNWARE_API_KEY:
                cell_states[i][0] = "<p style='color:orange; text-align:center;'>No Runware API key</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 4:  # SQZB 모델
            if SKIP_LOCAL_MODELS:
                cell_states[i][0] = "<p style='color:gray; text-align:center;'>Skipped (--skip-local)</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 5:  # WaveSpeed Fast
            if not WAVESPEED_API_KEY:
                cell_states[i][0] = "<p style='color:orange; text-align:center;'>No WaveSpeed API key</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 6:  # WaveSpeed Dev
            if not WAVESPEED_API_KEY:
                cell_states[i][0] = "<p style='color:orange; text-align:center;'>No WaveSpeed API key</p>"
            else:
                cell_states[i][0] = "<p style='text-align:center; font-weight:bold;'>준비 중...</p>"
        elif i == 7:  # Empty cell
            cell_states[i][0] = "<p style='text-align:center;'>Empty cell</p>"
    
    # Update UI
    log_debug("Updating cells to initial states...")
    try:
        outputs = yield_ui_update(cell_states)
        if outputs:
            log_debug(f"Yielding {len(outputs)} outputs")
            yield outputs
            log_debug("Cells updated successfully")
        else:
            log_debug("Failed to generate initial UI states")
            return
    except Exception as e:
        log_debug(f"Error updating cells: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        return
    
    # 각 API 서비스 및 로컬 모델에 대한 작업 시작
    api_futures = []
    
    # 로컬 모델 (vanilla 및 sqzb) 처리
    local_futures = []
    if not SKIP_LOCAL_MODELS:
        log_debug("Starting local model inference...")
        try:
            # Vanilla 모델 (right_future)
            cell_states[0][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
            # SQZB 모델 (left_future)
            cell_states[4][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
            
            # UI 업데이트
            outputs = yield_ui_update(cell_states)
            if outputs:
                yield outputs
                
            # 로컬 추론 시작 (ProcessPoolExecutor 사용하여 병렬 처리)
            if not SKIP_LOCAL_MODELS:
                # Vanilla 모델 (force_no_fp8=True)
                vanilla_future = _executor.submit(inference_worker, prompt, 0.0, USE_HPU_GRAPHS, shared_seed, force_no_fp8=True)
                local_futures.append((0, "Vanilla", vanilla_future))
                
                # SQZB 모델 (force_no_fp8=False)
                sqzb_future = _executor.submit(inference_worker, prompt, RDT, USE_HPU_GRAPHS, shared_seed, force_no_fp8=False)
                local_futures.append((4, "SQZB", sqzb_future))
        except Exception as e:
            log_debug(f"Error starting local model inference: {e}")
            log_debug(f"Traceback: {traceback.format_exc()}")
    
    # FAL.ai API
    if FAL_KEY:
        log_debug("Starting fal.ai API request")
        cell_states[1][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
        api_futures.append((1, "fal.ai", "fal"))
    
    # Replicate API 
    if REPLICATE_API_TOKEN:
        log_debug("Starting Replicate API request")
        cell_states[2][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
        api_futures.append((2, "Replicate", "replicate"))
    
    # Runware API
    if RUNWARE_API_KEY:
        log_debug("Starting Runware API request")  
        cell_states[3][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
        api_futures.append((3, "Runware", "runware"))
    
    # WaveSpeed Fast API
    if WAVESPEED_API_KEY:
        log_debug("Starting WaveSpeed Fast API request")
        cell_states[5][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
        api_futures.append((5, "WaveSpeed Fast", "wavespeed_fast"))
    
    # WaveSpeed Dev API
    if WAVESPEED_API_KEY:
        log_debug("Starting WaveSpeed Dev API request")
        cell_states[6][0] = "<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
        api_futures.append((6, "WaveSpeed Dev", "wavespeed_dev"))
    
    # UI 업데이트 (API 요청 및 로컬 모델 시작)
    outputs = yield_ui_update(cell_states)
    if outputs:
        yield outputs
    
    # 모든 API 요청 병렬로 실행
    import concurrent.futures
    import threading
    
    active_api_calls = len(api_futures)
    log_debug(f"Processing {active_api_calls} parallel API requests")
    
    # API 요청 비동기 처리를 위한 함수
    def process_api_request(idx, service_name, api_type):
        try:
            log_debug(f"Processing {service_name} request...")
            img = None
            elapsed = 0
            error = None
            
            # API 유형에 따라 적절한 함수 호출
            if api_type == "fal":
                img, elapsed, error = call_fal_api(prompt, seed=shared_seed)
            elif api_type == "replicate":
                img, elapsed, image_url = call_replicate_api(prompt, seed=shared_seed)
                if img is None:
                    error = "API call failed to return an image"
            elif api_type == "runware":
                img, elapsed = call_runware_api(prompt, seed=shared_seed)
                if img is None:
                    error = "API call failed to return an image"
            elif api_type.startswith("wavespeed"):
                model_type = "fast" if api_type == "wavespeed_fast" else "dev"
                img, elapsed = call_wavespeed_api(prompt, model_type=model_type, seed=shared_seed)
                if img is None:
                    error = "API call failed to return an image"
            
            # 결과 처리
            if error:
                log_debug(f"Error from {service_name} API: {error}")
                return idx, service_name, None, elapsed, error
            elif img:
                log_debug(f"{service_name} API call successful in {elapsed:.2f}s")
                return idx, service_name, img, elapsed, None
            else:
                log_debug(f"{service_name} API returned no image")
                return idx, service_name, None, elapsed, "No image was generated"
                
        except Exception as e:
            log_debug(f"Exception in {service_name} API request: {e}")
            log_debug(f"Traceback: {traceback.format_exc()}")
            return idx, service_name, None, 0, str(e)
    
    # 스레드풀을 사용해 API 요청 병렬 처리
    api_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(api_futures)) as executor:
        future_to_api = {executor.submit(process_api_request, idx, name, api_type): (idx, name) 
                         for idx, name, api_type in api_futures}
        
        for future in concurrent.futures.as_completed(future_to_api):
            idx, name = future_to_api[future]
            try:
                result = future.result()
                if result:
                    api_results.append(result)
                    # UI 업데이트
                    cell_idx, service_name, img, elapsed, error = result
                    if error:
                        cell_states[cell_idx][0] = f"<p style='color:red; text-align:center; font-size:0.8em;'>{str(error)[:50]}...</p>"
                    else:
                        cell_states[cell_idx][0] = f"<p style='color:green; text-align:center;'>Elapsed: {elapsed:.2f}s</p>"
                        cell_states[cell_idx][2] = img
                    
                    # 각 API 결과마다 UI 업데이트
                    outputs = yield_ui_update(cell_states)
                    if outputs:
                        yield outputs
            except Exception as e:
                log_debug(f"Error processing {name} API result: {e}")
                log_debug(f"Traceback: {traceback.format_exc()}")
                cell_states[idx][0] = f"<p style='color:red; text-align:center;'>{str(e)[:20]}...</p>"
                outputs = yield_ui_update(cell_states)
                if outputs:
                    yield outputs
    
    # 로컬 모델 결과 처리
    if local_futures:
        log_debug(f"Waiting for {len(local_futures)} local model inference results")
        
        for idx, name, future in local_futures:
            try:
                cell_states[idx][0] = f"<p style='text-align:center; font-weight:bold;'>Waiting...</p>"
                outputs = yield_ui_update(cell_states)
                if outputs:
                    yield outputs
                
                # 결과 대기 (timeout 60초)
                try:
                    result = future.result(timeout=60)
                    images, elapsed, seed = result
                    
                    if images:
                        # 이미지 처리
                        if isinstance(images, list) and len(images) > 0:
                            display_image = images[0]
                        else:
                            display_image = images
                        
                        cell_states[idx][0] = f"<p style='color:green; text-align:center;'>Elapsed: {elapsed:.2f}s</p>"
                        cell_states[idx][2] = display_image
                    else:
                        cell_states[idx][0] = f"<p style='color:orange; text-align:center;'>이미지 없음</p>"
                    
                    # UI 업데이트
                    outputs = yield_ui_update(cell_states)
                    if outputs:
                        yield outputs
                        
                except concurrent.futures.TimeoutError:
                    log_debug(f"{name} model inference timed out")
                    cell_states[idx][0] = f"<p style='color:orange; text-align:center;'>시간 초과</p>"
                    outputs = yield_ui_update(cell_states)
                    if outputs:
                        yield outputs
                
            except Exception as e:
                log_debug(f"Error processing {name} result: {e}")
                log_debug(f"Traceback: {traceback.format_exc()}")
                cell_states[idx][0] = f"<p style='color:red; text-align:center;'>오류: {str(e)[:50]}...</p>"
                outputs = yield_ui_update(cell_states)
                if outputs:
                    yield outputs
    
    # 최종 UI 업데이트
    log_debug("Sending final UI update")
    try:
        outputs = yield_ui_update(cell_states)
        if outputs:
            log_debug(f"Yielding final {len(outputs)} outputs")
            yield outputs
            log_debug("Final UI update successful")
        else:
            log_debug("Failed to generate final UI outputs")
    except Exception as e:
        log_debug(f"Error in final yield: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        return

def yield_ui_update(cell_states):
    """Helper function to yield UI updates from the cell states"""
    try:
        outputs = []
        for i, cell in enumerate(cell_states):
            outputs.append(cell[0])                     # time_html content
            outputs.append(cell[1])                     # progress_html
            
            # Show image if available
            if cell[2] is not None:                     # If we have an image
                outputs.append(gr.update(visible=True, value=cell[2]))
            else:
                outputs.append(gr.update(visible=False))
        
        log_debug(f"Yielding {len(outputs)} outputs")
        return outputs  # Return the outputs instead of yielding
    except Exception as e:
        log_debug(f"Error creating UI update: {str(e)}")
        log_debug(f"Traceback: {traceback.format_exc()}")
        return None

with gr.Blocks(title="Flux Image Generation UI", css=custom_css) as demo:
    # Main title at the top
    if args.compare:
        gr.Markdown(f"# Flux.1 API Comparison ({HEIGHT}x{WIDTH} resolution)", elem_classes=["main-title"])
    else:
        gr.Markdown(f"# Flux.1 [dev] Image Generation Demo ({HEIGHT}x{WIDTH} resolution)", elem_classes=["main-title"])
    
    if args.compare:
        # Compare mode layout (2x4 grid)
        # Row 1: [right_future, fal, replicate, runware]
        with gr.Row(elem_classes=["contain-row"]) as top_row:
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col1:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## Vanilla version", elem_classes=["title-text"], )
                    top_time1 = gr.HTML("Loading...", elem_classes=["time-text"])
                top_progress1 = gr.HTML(visible=False)
                top_image1 = gr.Image(label="Vanilla", visible=False, elem_classes=["large-image"], height=250, width=250)
                
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col2:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## Fal.ai", elem_classes=["title-text"])
                    top_time2 = gr.HTML("Loading...", elem_classes=["time-text"])
                top_progress2 = gr.HTML(visible=False)
                top_image2 = gr.Image(label="Fal.ai", visible=False, elem_classes=["large-image"], height=250, width=250)
                
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col3:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## Replicate", elem_classes=["title-text"])
                    top_time3 = gr.HTML("Loading...", elem_classes=["time-text"])
                top_progress3 = gr.HTML(visible=False)
                top_image3 = gr.Image(label="Replicate", visible=False, elem_classes=["large-image"], height=250, width=250)
                
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col4:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## Runware", elem_classes=["title-text"])
                    top_time4 = gr.HTML("Loading...", elem_classes=["time-text"])
                top_progress4 = gr.HTML(visible=False)
                top_image4 = gr.Image(label="Runware", visible=False, elem_classes=["large-image"], height=250, width=250)
        
        # Row 2: [left_future, wavespeed_fast, wavespeed_dev, empty]
        with gr.Row(elem_classes=["contain-row"]) as bottom_row:
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col5:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## SQZB Optimized", elem_classes=["title-text"])
                    bottom_time1 = gr.HTML("Loading...", elem_classes=["time-text"])
                bottom_progress1 = gr.HTML(visible=False)
                bottom_image1 = gr.Image(label="SQZB", visible=False, elem_classes=["large-image"], height=250, width=250)
                
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col6:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## WaveSpeed Fast", elem_classes=["title-text"])
                    bottom_time2 = gr.HTML("Loading...", elem_classes=["time-text"])
                bottom_progress2 = gr.HTML(visible=False)
                bottom_image2 = gr.Image(label="WaveSpeed Fast", visible=False, elem_classes=["large-image"], height=250, width=250)
                
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col7:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## WaveSpeed Dev", elem_classes=["title-text"])
                    bottom_time3 = gr.HTML("Loading...", elem_classes=["time-text"])
                bottom_progress3 = gr.HTML(visible=False)
                bottom_image3 = gr.Image(label="WaveSpeed Dev", visible=False, elem_classes=["large-image"], height=250, width=250)
                
            with gr.Column(scale=1, elem_classes=["compact-column"]) as col8:
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("## (Empty)", elem_classes=["title-text"])
                    bottom_time4 = gr.HTML("", elem_classes=["time-text"])
                bottom_progress4 = gr.HTML(visible=False)
                bottom_image4 = gr.Image(label="Empty", visible=False, elem_classes=["large-image"], height=250, width=250)
    else:
        # Original layout (1x2 grid)
        with gr.Row(elem_classes=["contain-row"]):
            with gr.Column() as left_column:
                # Left column title
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("# Intel Gaudi-2 HPU 🤝 SqueezeBits (optimized)", elem_classes=["title-text"])
                    left_time = gr.HTML("Elapsed: ... seconds", elem_classes=["time-text"])
                
                # Replace gallery with image component for direct full-size display
                left_image = gr.Image(
                    label="SQZB Result", 
                    visible=False,
                    elem_id="left-image",
                    elem_classes=["large-image"],
                    height=640,
                    show_download_button=True
                )
                
                # 아이콘 대신 프로그레스 바로 변경
                left_progress_display = gr.HTML(visible=False)
                
            with gr.Column() as right_column:
                # Right column title
                with gr.Row(elem_classes=["title-container"]):
                    gr.Markdown("# Vanilla version <br>(not optimized)", elem_classes=["title-text"])
                    right_time = gr.HTML("Elapsed: ... seconds", elem_classes=["time-text"])
                
                # Replace gallery with image component for direct full-size display
                right_image = gr.Image(
                    label="Vanilla Result",
                    visible=False,
                    elem_id="right-image",
                    elem_classes=["large-image"],
                    height=640,
                    show_download_button=True
                )
                
                # 아이콘 대신 프로그레스 바로 변경
                right_progress_display = gr.HTML(visible=False)
    
    # Move prompt area here - immediately after the galleries
    prompt_box = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=2)

    with gr.Row():
        with gr.Column(scale=2.5, elem_id="random-seed-checkbox"):
            seed_checkbox = gr.Checkbox(label="Use Random Seed", value=False)
        with gr.Column(scale=3.5):
            # Replace standard Number component with a Row containing Text and Number
            with gr.Row(elem_id="seed-row"):
                gr.Markdown("Seed(0 to 4294967295):", elem_classes=["seed-label"])
                seed_input = gr.Number(
                    label="", 
                    value=42, 
                    minimum=0, 
                    maximum=2**32-1, 
                    step=1, 
                    precision=0, 
                    interactive=True,  # Make it interactive by default and let the toggle function handle it
                    container=False
                )
        with gr.Column(scale=4):
            sample_prompt_btn = gr.Button("Use Sample Prompt")
        with gr.Column(scale=4):
            submit_btn = gr.Button("Generate", variant="primary")

    # Update seed input interactivity based on checkbox
    def toggle_seed_input(use_random):
        return gr.update(interactive=not use_random)
    
    seed_checkbox.change(fn=toggle_seed_input, inputs=[seed_checkbox], outputs=[seed_input])
    
    def on_click_sample_prompt():
        return gr.update(value=random.choice(SAMPLE_PROMPTS))

    if args.compare:
        # Connect submit button for compare mode
        compare_outputs = [
            # Top row outputs
            top_time1, top_progress1, top_image1,
            top_time2, top_progress2, top_image2,
            top_time3, top_progress3, top_image3,
            top_time4, top_progress4, top_image4,
            # Bottom row outputs
            bottom_time1, bottom_progress1, bottom_image1,
            bottom_time2, bottom_progress2, bottom_image2,
            bottom_time3, bottom_progress3, bottom_image3,
            bottom_time4, bottom_progress4, bottom_image4,
        ]
        print(f"Setting up compare mode interface with {len(compare_outputs)} outputs")
        
        # Special debug message
        log_debug("COMPARE MODE EVENT HANDLERS SETUP")
        log_debug(f"Number of outputs: {len(compare_outputs)}")
        log_debug(f"Using on_submit_generate_compare function")
        
        submit_btn.click(
            fn=on_submit_generate_compare,
            inputs=[prompt_box, seed_input],
            outputs=compare_outputs,
        )
        
        prompt_box.submit(
            fn=on_submit_generate_compare,
            inputs=[prompt_box, seed_input],
            outputs=compare_outputs,
        )
        
        log_debug("Event handlers setup complete")
    else:
        # Connect submit button for original mode
        print("Setting up standard mode interface")
        standard_outputs = [left_time, left_progress_display, left_image, right_time, right_progress_display, right_image]
        submit_btn.click(
            on_submit_generate,
            inputs=[prompt_box, seed_input, seed_checkbox],
            outputs=standard_outputs,
        )
        prompt_box.submit(
            on_submit_generate,
            inputs=[prompt_box, seed_input, seed_checkbox],
            outputs=standard_outputs,
        )
    
    sample_prompt_btn.click(fn=on_click_sample_prompt, inputs=None, outputs=prompt_box)

# Launch the app
if args.compare:
    print("Starting in API comparison mode with 2x4 grid layout")
else:
    print("Starting in standard mode with 1x2 grid layout")

demo.launch(share=True) 