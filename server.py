#!/usr/bin/env python3
"""
Flux Studio - Multi-Backend Image Generation Server
Supports: Native Diffusers, Ollama (Mac), Stable Diffusion WebUI, ComfyUI
"""

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
import base64
import os
import re
import io
import threading
from PIL import Image

# ============ CONFIGURATION ============
PORT = 6010

# Native Diffusers pipeline (loaded on first use)
diffusers_pipeline = None
diffusers_model_name = None
diffusers_lock = threading.Lock()

# Available native models (fully open - no auth required)
NATIVE_MODELS = [
    "Lykon/dreamshaper-8",
    "nitrosocke/Arcane-Diffusion",
    "prompthero/openjourney-v4",
    "SimianLuo/LCM_Dreamshaper_v7",
    "segmind/tiny-sd",
]

# Backend URLs (all local)
BACKENDS = {
    "native": {
        "name": "Native (Diffusers)",
        "url": None,
        "api_path": None,
        "models_path": None,
        "description": "Built-in - no setup required"
    },
    "ollama": {
        "name": "Ollama",
        "url": "http://localhost:11434",
        "api_path": "/api/generate",
        "models_path": "/api/tags",
        "description": "Ollama (Mac M1/M2/M3 only)"
    },
    "sd-webui": {
        "name": "Stable Diffusion WebUI",
        "url": "http://localhost:7860",
        "api_path": "/sdapi/v1/txt2img",
        "models_path": "/sdapi/v1/sd-models",
        "description": "AUTOMATIC1111 WebUI (Windows/Linux)"
    },
    "comfyui": {
        "name": "ComfyUI", 
        "url": "http://localhost:8188",
        "api_path": "/prompt",
        "models_path": "/object_info",
        "description": "ComfyUI (Windows/Linux)"
    }
}

# Default Ollama models (for Mac)
OLLAMA_IMAGE_MODELS = [
    "x/z-image-turbo:bf16",
    "x/z-image-turbo:fp8",
    "x/z-image-turbo:latest",
    "x/flux2-klein:4b",
    "x/flux2-klein:9b",
    "x/flux2-klein:latest"
]

HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
OLLAMA_LOG_LOCATION = "~/.ollama/logs/server.log"
LOG_MESSAGE_PATTERN = r'msg="([^"]*)"'

# ============ HELPER FUNCTIONS ============

def get_latest_log_message(level):
    """Return the latest log message for a given level, or None if not found."""
    log_path = Path(OLLAMA_LOG_LOCATION).expanduser()
    if not log_path.exists():
        return None
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    pattern = re.compile(rf"level={re.escape(level)}[^\n]*?{LOG_MESSAGE_PATTERN}")
    matches = list(pattern.finditer(log_text))
    if not matches:
        return None
    return matches[-1].group(1)


def ensure_history_dir():
    """Ensure the history directory exists"""
    HISTORY_DIR.mkdir(exist_ok=True)


def parse_history_id(path):
    """Extract and validate a history id from a request path."""
    if not path.startswith("/history/"):
        return None
    image_name = path.split("/history/", 1)[1]
    if image_name.endswith(".png"):
        image_name = image_name[:-4]
    if not image_name or not HISTORY_ID_PATTERN.fullmatch(image_name):
        return None
    return image_name


def resolve_history_path(image_name, suffix):
    """Resolve a history file path and ensure it stays under HISTORY_DIR."""
    base_dir = HISTORY_DIR.resolve()
    candidate = (base_dir / f"{image_name}{suffix}").resolve()
    if base_dir not in candidate.parents:
        return None
    return candidate


def check_backend_status(backend_id):
    """Check if a backend is available"""
    if backend_id not in BACKENDS:
        return False
    
    # Native backend is always available
    if backend_id == "native":
        return True
    
    backend = BACKENDS[backend_id]
    if not backend.get("url"):
        return False
    try:
        req = urllib.request.Request(backend["url"], method="GET")
        req.add_header("User-Agent", "FluxStudio/1.0")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except:
        return False


def get_native_models():
    """Return available native diffusers models"""
    return NATIVE_MODELS.copy()


def warmup_models():
    """Preload and warm up available AI models on startup"""
    print("\n🔥 Warming up AI models...")
    
    # Warm up native diffusers if available
    try:
        print("   Loading native diffusers pipeline...")
        # Load the default model to cache it
        load_diffusers_pipeline(NATIVE_MODELS[0])
        print("   ✅ Native diffusers ready")
    except Exception as e:
        print(f"   ⚠️ Native warmup skipped: {e}")
    
    # Check Ollama
    if check_backend_status("ollama"):
        print("   ✅ Ollama detected and ready")
    
    # Check SD WebUI
    if check_backend_status("sd-webui"):
        print("   ✅ SD WebUI detected and ready")
    
    # Check ComfyUI
    if check_backend_status("comfyui"):
        print("   ✅ ComfyUI detected and ready")
    
    print("🚀 Warmup complete!\n")


def load_diffusers_pipeline(model_name):
    """Load or reload the diffusers pipeline"""
    global diffusers_pipeline, diffusers_model_name
    
    with diffusers_lock:
        if diffusers_pipeline is not None and diffusers_model_name == model_name:
            return diffusers_pipeline
        
        print(f"🔄 Loading model: {model_name}")
        print("   (This may take a while on first run - downloading model...)")
        
        try:
            import torch
            # Use StableDiffusionPipeline directly to avoid AutoPipeline import issues
            from diffusers import StableDiffusionPipeline, LCMScheduler
            
            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                print("   Using CUDA (GPU)")
            else:
                device = "cpu"
                dtype = torch.float32
                print("   Using CPU (slower)")
            
            # Check if it's an LCM model (faster, fewer steps needed)
            is_lcm = "lcm" in model_name.lower()
            
            # Load pipeline
            diffusers_pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            
            # Use LCM scheduler for LCM models
            if is_lcm:
                diffusers_pipeline.scheduler = LCMScheduler.from_config(
                    diffusers_pipeline.scheduler.config
                )
            
            diffusers_pipeline = diffusers_pipeline.to(device)
            
            # Enable optimizations for local/fast inference
            if device == "cuda":
                # Enable memory efficient attention if available
                try:
                    diffusers_pipeline.enable_xformers_memory_efficient_attention()
                    print("   ✅ xFormers memory efficient attention enabled")
                except:
                    try:
                        diffusers_pipeline.enable_attention_slicing(1)
                    except:
                        pass
                
                # Enable VAE slicing for lower VRAM
                try:
                    diffusers_pipeline.enable_vae_slicing()
                except:
                    pass
            
            # Set to eval mode for faster inference
            diffusers_pipeline.unet.eval()
            diffusers_pipeline.vae.eval()
            
            diffusers_model_name = model_name
            print(f"✅ Model loaded: {model_name}")
            return diffusers_pipeline
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise


def generate_with_native(request_data, progress_callback=None):
    """Generate image using native diffusers pipeline with live preview"""
    import torch
    
    model = request_data.get("model", NATIVE_MODELS[0])
    prompt = request_data.get("prompt", "")
    negative_prompt = request_data.get("negative_prompt", "")
    width = request_data.get("width", 512)
    height = request_data.get("height", 512)
    steps = request_data.get("steps", 20)
    cfg_scale = request_data.get("cfg_scale", 7.0)
    seed = request_data.get("seed", -1)
    preview_interval = 3  # Send preview every 3 steps for faster response
    
    # Load pipeline
    pipe = load_diffusers_pipeline(model)
    
    # Set seed
    generator = None
    if seed != -1:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Helper to decode latents to preview image
    def latents_to_preview(latents):
        """Convert latents to a small preview image"""
        try:
            # Decode latents to image
            with torch.no_grad():
                # Scale latents
                latents_scaled = latents / pipe.vae.config.scaling_factor
                # Decode
                image = pipe.vae.decode(latents_scaled, return_dict=False)[0]
                # Convert to PIL
                image = pipe.image_processor.postprocess(image, output_type="pil")[0]
                
                # Resize to smaller preview (faster transfer)
                preview_size = (192, 192)  # Reduced from 256x256
                image.thumbnail(preview_size, Image.Resampling.BILINEAR)
                
                # Convert to base64 with lower quality
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=50, optimize=True)
                return base64.b64encode(buffer.getvalue()).decode("utf-8")
        except Exception as e:
            return None
    
    # Progress callback with preview
    def callback_fn(pipe, step, timestep, callback_kwargs):
        if progress_callback:
            # Get preview image at intervals
            preview = None
            if step % preview_interval == 0 or step == steps - 1:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    preview = latents_to_preview(latents)
            
            # Send step+1 because step is 0-indexed but display should be 1-indexed
            progress_callback(step + 1, steps, preview)
        return callback_kwargs
    
    # Generate
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        generator=generator,
        callback_on_step_end=callback_fn,
    )
    
    # Convert to base64
    image = result.images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True, compress_level=6)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return image_base64


def get_sd_webui_models():
    """Get available models from Stable Diffusion WebUI"""
    try:
        req = urllib.request.Request(
            BACKENDS["sd-webui"]["url"] + "/sdapi/v1/sd-models",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
            return [m.get("model_name", m.get("title", "unknown")) for m in models]
    except Exception as e:
        print(f"Error fetching SD WebUI models: {e}")
        return []


def get_sd_webui_samplers():
    """Get available samplers from SD WebUI"""
    try:
        req = urllib.request.Request(
            BACKENDS["sd-webui"]["url"] + "/sdapi/v1/samplers",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            samplers = json.loads(response.read().decode("utf-8"))
            return [s.get("name") for s in samplers]
    except:
        return ["Euler a", "DPM++ 2M Karras", "DDIM"]


def get_ollama_models():
    """Get available image generation models from Ollama"""
    try:
        req = urllib.request.Request(
            BACKENDS["ollama"]["url"] + "/api/tags",
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            models = []
            for model in data.get("models", []):
                if model.get("name") in OLLAMA_IMAGE_MODELS:
                    models.append(model["name"])
            return models
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []


def generate_with_sd_webui(request_data):
    """Generate image using Stable Diffusion WebUI API"""
    # Build the SD WebUI API request
    sd_request = {
        "prompt": request_data.get("prompt", ""),
        "negative_prompt": request_data.get("negative_prompt", ""),
        "width": request_data.get("width", 512),
        "height": request_data.get("height", 512),
        "steps": request_data.get("steps", 20),
        "seed": request_data.get("seed", -1),
        "cfg_scale": request_data.get("cfg_scale", 7),
        "sampler_name": request_data.get("sampler", "Euler a"),
        "batch_size": 1,
        "n_iter": 1,
    }
    
    req = urllib.request.Request(
        BACKENDS["sd-webui"]["url"] + "/sdapi/v1/txt2img",
        data=json.dumps(sd_request).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
        if result.get("images") and len(result["images"]) > 0:
            return result["images"][0]  # Base64 encoded image
    return None


def generate_with_ollama(request_data, handler):
    """Generate image using Ollama API (streams progress)"""
    reformatted_body = json.dumps(request_data)
    
    req = urllib.request.Request(
        BACKENDS["ollama"]["url"] + "/api/generate",
        data=reformatted_body.encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as response:
        handler.send_response(response.status)
        content_type = response.headers.get("Content-Type", "application/x-ndjson")
        handler.send_header("Content-Type", content_type)
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("X-Accel-Buffering", "no")
        handler.end_headers()
        
        final_image_data = None
        while True:
            line = response.readline()
            if not line:
                break
            try:
                line_data = json.loads(line)
                if line_data.get('done') and line_data.get('image'):
                    final_image_data = line_data.get('image')
            except:
                pass
            handler.wfile.write(line)
            handler.wfile.flush()
        
        return final_image_data


def save_to_history(image_base64, request_data, backend):
    """Save generated image to history"""
    try:
        ensure_history_dir()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Save image
        image_path = HISTORY_DIR / f"{timestamp}.png"
        with open(image_path, 'wb') as f:
            f.write(base64.b64decode(image_base64))
        
        # Save metadata
        metadata = {
            "id": timestamp,
            "timestamp": datetime.now().isoformat(),
            "backend": backend,
            "settings": {
                "model": request_data.get("model", ""),
                "prompt": request_data.get("prompt", ""),
                "negative_prompt": request_data.get("negative_prompt", ""),
                "seed": request_data.get("seed", 0),
                "width": request_data.get("width", 512),
                "height": request_data.get("height", 512),
                "steps": request_data.get("steps", 20),
                "cfg_scale": request_data.get("cfg_scale", 7),
                "sampler": request_data.get("sampler", ""),
            }
        }
        
        json_path = HISTORY_DIR / f"{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved image to history: {timestamp}.png")
        return timestamp
    except Exception as e:
        print(f"Error saving to history: {e}")
        return None


# ============ REQUEST HANDLER ============

class ProxyHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/" or self.path == "/index.html":
            self.serve_file("index.html", "text/html")
        
        elif self.path == "/backends":
            # Return available backends and their status
            result = {}
            for backend_id, backend in BACKENDS.items():
                result[backend_id] = {
                    "name": backend["name"],
                    "description": backend["description"],
                    "available": check_backend_status(backend_id)
                }
            self.send_json(result)
        
        elif self.path.startswith("/models"):
            # Get models based on query param or detect available backend
            backend = self.get_query_param("backend")
            models = []
            
            if backend == "native":
                models = get_native_models()
            elif backend == "sd-webui" or (not backend and check_backend_status("sd-webui")):
                models = get_sd_webui_models()
            elif backend == "ollama" or (not backend and check_backend_status("ollama")):
                models = get_ollama_models()
            elif not backend:
                # Default to native if nothing else specified
                models = get_native_models()
            
            self.send_json(models)
        
        elif self.path == "/samplers":
            # Get SD WebUI samplers
            samplers = get_sd_webui_samplers()
            self.send_json(samplers)
        
        elif self.path == "/ollamalog/warn":
            message = get_latest_log_message("WARN")
            if message:
                self.send_json({"message": message})
            else:
                self.send_error_json(404, "No WARN entry found")
        
        elif self.path == "/ollamalog/info":
            message = get_latest_log_message("INFO")
            if message:
                self.send_json({"message": message})
            else:
                self.send_error_json(404, "No INFO entry found")
        
        elif self.path == "/history/index":
            self.get_history_index()
        
        elif self.path.startswith("/history/"):
            self.get_history_item()
        
        else:
            self.send_error(404, "File not found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/generate":
            self.handle_generate()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_DELETE(self):
        """Handle DELETE requests"""
        if self.path.startswith("/history/"):
            self.delete_history_item()
        else:
            self.send_error(404, "Endpoint not found")
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    # ---- Helper Methods ----
    
    def get_query_param(self, name):
        """Extract query parameter from path"""
        if "?" in self.path:
            query = self.path.split("?", 1)[1]
            for param in query.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    if key == name:
                        return value
        return None
    
    def serve_file(self, filename, content_type):
        """Serve a static file"""
        try:
            file_path = Path(__file__).parent / filename
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, f"{filename} not found")
    
    def send_json(self, data):
        """Send JSON response"""
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def send_error_json(self, code, message):
        """Send error as JSON"""
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
    
    # ---- Generate Handler ----
    
    def handle_generate(self):
        """Handle image generation request"""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            request_data = json.loads(body)
            
            print(f"\n📥 Generate request:")
            print(json.dumps(request_data, indent=2))
            
            backend = request_data.get("backend", "auto")
            
            # Auto-detect backend
            if backend == "auto":
                if check_backend_status("native"):
                    backend = "native"
                elif check_backend_status("sd-webui"):
                    backend = "sd-webui"
                elif check_backend_status("ollama"):
                    backend = "ollama"
                else:
                    self.send_error_json(503, "No image generation backend available.")
                    return
            
            print(f"🎨 Using backend: {backend}")
            
            # Clean prompt
            request_data['prompt'] = request_data.get('prompt', '').replace('\n', ' ').replace('\r', ' ')
            
            if backend == "native":
                self.generate_native(request_data)
            elif backend == "sd-webui":
                self.generate_sd_webui(request_data)
            elif backend == "ollama":
                self.generate_ollama(request_data)
            else:
                self.send_error_json(400, f"Unknown backend: {backend}")
                
        except json.JSONDecodeError:
            self.send_error_json(400, "Invalid JSON body")
        except Exception as e:
            print(f"❌ Error: {e}")
            self.send_error_json(500, str(e))
    
    def generate_native(self, request_data):
        """Generate with native Diffusers pipeline"""
        try:
            # Send initial progress
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            
            # Send "generating" status immediately
            self.wfile.write(json.dumps({"status": "generating", "backend": "native"}).encode() + b"\n")
            self.wfile.flush()
            
            # Progress callback with preview support
            def progress_callback(step, total, preview=None):
                progress_data = {"completed": step, "total": total}
                if preview:
                    progress_data["preview"] = preview
                try:
                    self.wfile.write(json.dumps(progress_data).encode() + b"\n")
                    self.wfile.flush()
                except:
                    pass
            
            # Generate
            image_base64 = generate_with_native(request_data, progress_callback)
            
            if image_base64:
                # Save to history
                save_to_history(image_base64, request_data, "native")
                
                # Send result
                result = {"done": True, "image": image_base64}
                self.wfile.write(json.dumps(result).encode() + b"\n")
                self.wfile.flush()
                print("✅ Image generated successfully")
            else:
                self.wfile.write(json.dumps({"error": "No image returned"}).encode() + b"\n")
                
        except Exception as e:
            print(f"❌ Native generation error: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.wfile.write(json.dumps({"error": str(e)}).encode() + b"\n")
            except:
                pass
    
    def generate_sd_webui(self, request_data):
        """Generate with Stable Diffusion WebUI"""
        try:
            # Send initial progress
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            
            # Send "generating" status
            self.wfile.write(json.dumps({"status": "generating", "backend": "sd-webui"}).encode() + b"\n")
            self.wfile.flush()
            
            # Generate
            image_base64 = generate_with_sd_webui(request_data)
            
            if image_base64:
                # Save to history
                save_to_history(image_base64, request_data, "sd-webui")
                
                # Send result
                result = {"done": True, "image": image_base64}
                self.wfile.write(json.dumps(result).encode() + b"\n")
                self.wfile.flush()
                print("✅ Image generated successfully")
            else:
                self.wfile.write(json.dumps({"error": "No image returned"}).encode() + b"\n")
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else str(e)
            self.wfile.write(json.dumps({"error": f"SD WebUI error: {error_body}"}).encode() + b"\n")
        except urllib.error.URLError as e:
            self.wfile.write(json.dumps({"error": f"Cannot connect to SD WebUI: {e}"}).encode() + b"\n")
    
    def generate_ollama(self, request_data):
        """Generate with Ollama"""
        try:
            image_data = generate_with_ollama(request_data, self)
            if image_data:
                save_to_history(image_data, request_data, "ollama")
                print("✅ Image generated successfully")
        except urllib.error.HTTPError as e:
            error_body = e.read() if e.fp else b""
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(error_body or json.dumps({"error": str(e)}).encode())
        except urllib.error.URLError as e:
            self.send_error_json(502, f"Cannot connect to Ollama: {e}")
    
    # ---- History Handlers ----
    
    def get_history_index(self):
        """Return list of all history items"""
        try:
            ensure_history_dir()
            history_items = []
            
            for json_file in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
                try:
                    with open(json_file, 'r') as f:
                        item_data = json.load(f)
                        item_data.pop("image", None)  # Don't include full image
                        history_items.append(item_data)
                except:
                    continue
            
            self.send_json(history_items)
        except Exception as e:
            self.send_error_json(500, f"Failed to load history: {e}")
    
    def get_history_item(self):
        """Return a specific history item with image"""
        image_name = parse_history_id(self.path)
        if not image_name:
            self.send_error_json(400, "Invalid history id")
            return
        
        try:
            ensure_history_dir()
            image_path = resolve_history_path(image_name, ".png")
            json_path = resolve_history_path(image_name, ".json")
            
            if not image_path or not json_path:
                self.send_error_json(400, "Invalid history path")
                return
            
            if not image_path.exists():
                self.send_error_json(404, f"Image not found: {image_name}")
                return
            
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            metadata = {}
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
            
            response_data = {
                "image": image_data,
                "settings": metadata.get("settings", {}),
                "timestamp": metadata.get("timestamp", ""),
                "id": metadata.get("id", image_name),
                "backend": metadata.get("backend", "unknown")
            }
            
            self.send_json(response_data)
        except Exception as e:
            self.send_error_json(500, f"Failed to load image: {e}")
    
    def delete_history_item(self):
        """Delete a history item"""
        image_name = parse_history_id(self.path)
        if not image_name:
            self.send_json({"SUCCESS": False, "error": "Invalid history id"})
            return
        
        try:
            ensure_history_dir()
            image_path = resolve_history_path(image_name, ".png")
            json_path = resolve_history_path(image_name, ".json")
            
            if not image_path or not json_path:
                self.send_json({"SUCCESS": False, "error": "Invalid history path"})
                return
            
            if not image_path.exists() and not json_path.exists():
                self.send_json({"SUCCESS": False, "error": f"Image not found: {image_name}"})
                return
            
            if image_path.exists():
                image_path.unlink()
            if json_path.exists():
                json_path.unlink()
            
            self.send_json({"SUCCESS": True})
            print(f"✓ Deleted history item: {image_name}")
        except Exception as e:
            self.send_json({"SUCCESS": False, "error": str(e)})
    
    def log_message(self, format, *args):
        """Custom log format"""
        print(f"[{self.log_date_time_string()}] {format % args}")


# ============ MAIN ============

def main():
    ensure_history_dir()
    
    # Check which backends are available
    print("🔍 Checking available backends...")
    for backend_id, backend in BACKENDS.items():
        available = check_backend_status(backend_id)
        status = "✅ Available" if available else "❌ Not running"
        print(f"   {backend['name']}: {status}")
    
    # Warm up models for faster first generation
    warmup_models()
    
    server_address = ("", PORT)
    httpd = ThreadingHTTPServer(server_address, ProxyHandler)
    
    print(f"🚀 DreamForge - Optimized Local AI Image Generator")
    print(f"📡 Server running on http://localhost:{PORT}")
    print(f"🎨 Backends: Native (Diffusers), Ollama, SD WebUI, ComfyUI")
    print(f"💡 Models pre-loaded and ready for instant generation!")
    print(f"⚡ Optimized for local performance with GPU acceleration")
    print(f"Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\n👋 Shutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
