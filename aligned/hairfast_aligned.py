"""
HairFastGAN Standalone API (Article Mode)
This script deploys a high-fidelity hairstyle transfer API on Modal.
It uses the 'Article' version weights and includes Poisson blending for realism.
"""

import modal
import io
import base64
from typing import Optional
from fastapi import UploadFile, File

# 1. DEFINE THE MODAL APP
# 'hairfast-api-exact' is the name that will appear in your Modal dashboard.
app = modal.App("hairfast-api-exact")

# 2. CONFIGURE THE CONTAINER IMAGE
# We start with a CUDA-enabled Ubuntu base to support GPU operations.
image = (
    modal.Image.from_registry("nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    # Install system-level dependencies for OpenCV, Dlib, and image processing.
    .apt_install(
        "git", "libglib2.0-0", "libsm6", "libxext6", 
        "libxrender-dev", "libgomp1", "libgl1-mesa-glx", "cmake", "unzip", "wget"
    )
    # Download and install 'ninja' - a build system required to compile StyleGAN2 custom CUDA kernels at runtime.
    .run_commands(
        "wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip",
        "unzip ninja-linux.zip -d /usr/local/bin/",
        "update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force"
    )
    # Install Python packages. Note specific versions for torch/torchvision to match StyleGAN2 compatibility.
    .pip_install(
        "fastapi[standard]", "gdown", "torch==1.13.1", "torchvision==0.14.1",
        "ninja==1.11.1", "Pillow==10.0.0", "numpy==1.23.5", "scipy==1.10.1",
        "opencv-python==4.8.0.74", "scikit-image==0.21.0", "tqdm==4.65.0",
        "matplotlib==3.7.2", "face-alignment", "dill==0.2.7.1", "addict",
        "fpie",      # Fast Poisson Image Editing for seamless hair-to-skin transitions.
        "dlib-bin"   # Pre-compiled dlib for faster builds and better reliability.
    )
    # CLIP is used by HairFastGAN for color alignment and feature extraction.
    .pip_install("git+https://github.com/openai/CLIP.git")
    # Clone the source code repository into the container.
    .run_commands(
        "git clone https://github.com/AIRI-Institute/HairFastGAN /root/HairFastGAN"
    )
    # BAKE MODELS: This takes your local 'pretrained_models' folder (which should contain ~127MB ffhq.pt)
    # and mounts it inside the container exactly where the code expects to find weights.
    .add_local_dir("./pretrained_models", remote_path="/root/HairFastGAN/pretrained_models")
)

# 3. DEFINE THE MODEL CLASS
# We use @app.cls to wrap the model loading and inference logic, allowing it to stay 'warm' on a GPU.
@app.cls(
    image=image, 
    gpu="T4",           # Using an NVIDIA T4 GPU (cost-effective for this model).
    timeout=600,        # Give the model up to 10 minutes to process (prevents kills on cold starts).
    scaledown_window=300 # Keep the GPU warm for 5 minutes after the last request.
)
class HairFastModel:
    @modal.enter()
    def load_model(self):
        """
        Runs once when the container starts. 
        Pre-loads the weights into GPU memory so requests are fast.
        """
        import sys, os
        # Change working directory so relative imports inside HairFastGAN work.
        os.chdir('/root/HairFastGAN')
        sys.path.insert(0, '/root/HairFastGAN')
        
        # Set CUDA_HOME so the Ninja compiler knows where to find NVIDIA tools.
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        
        from hair_swap import HairFast, get_parser
        
        # get_parser().parse_args([]) initializes the 'Article' version of the model by default.
        # This includes Pose, Shape, and Color Alignment modules.
        self.hair_fast = HairFast(get_parser().parse_args([]))

    @modal.method()
    def inference(self, face_bytes: bytes, shape_bytes: bytes, color_bytes: Optional[bytes] = None):
        """
        The core logic for swapping hair.
        """
        from PIL import Image
        import io, torch
        import torchvision.transforms as T
        from utils.image_utils import poisson_image_blending
        
        # Convert raw bytes from the API request into PIL Images for processing.
        face_img = Image.open(io.BytesIO(face_bytes)).convert('RGB')
        shape_img = Image.open(io.BytesIO(shape_bytes)).convert('RGB')
        
        # Color Reference Logic:
        # If the user provides a color image, we use it. 
        # Otherwise, we extract color from the hairstyle reference (shape_img).
        color_img = Image.open(io.BytesIO(color_bytes)).convert('RGB') if color_bytes else shape_img
        
        # swap() performs the multi-stage alignment.
        # align=True is critical; it uses landmarks to perfectly fit the hair to the input face's pose.
        result, face_align, _, _ = self.hair_fast.swap(face_img, shape_img, color_img, align=True)
        
        # POISSON BLENDING:
        # This smooths the border where the new hair meets the forehead.
        # dilate_erosion=15 is the "sweet spot" found in the original research for natural results.
        result, _ = poisson_image_blending(result, face_align, dilate_erosion=15)
        
        # Some post-processing steps might return a Torch Tensor. 
        # We ensure the final output is a PIL Image before saving to the buffer.
        if not hasattr(result, 'save'):
            result = T.functional.to_pil_image(result)
            
        # Convert the final Image back to bytes to send across the network.
        buf = io.BytesIO()
        result.save(buf, format='PNG')
        return buf.getvalue()

# 4. DEFINE THE WEB ENDPOINT
# This creates a public URL that accepts POST requests.
@app.function(image=image)
@modal.web_endpoint(method="POST")
async def transfer_hair(
    face_image: UploadFile = File(...),   # Primary person image
    shape_image: UploadFile = File(...),  # Hairstyle reference image
    color_image: Optional[UploadFile] = File(None) # Optional color reference
):
    """
    HTTP Handler: Receives files, triggers GPU inference, and returns base64 results.
    """
    try:
        # Read files into memory.
        face_bytes = await face_image.read()
        shape_bytes = await shape_image.read()
        color_bytes = await color_image.read() if color_image else None
        
        # Instantiate the Model Class (Modal handles the GPU scaling automatically).
        model = HairFastModel()
        
        # Run inference on the GPU.
        res_bytes = model.inference.remote(face_bytes, shape_bytes, color_bytes)
        
        # Encode as Base64 so it can be easily returned in a JSON response.
        return {
            "success": True, 
            "result_image": base64.b64encode(res_bytes).decode('utf-8')
        }
    except Exception as e:
        # Return error details if something fails (e.g., face not detected).
        return {"success": False, "error": str(e)}