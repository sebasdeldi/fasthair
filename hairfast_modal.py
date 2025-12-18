"""
HairFastGAN Modal Deployment - Simplified
A web API for virtual hairstyle transfer using HairFastGAN
Note: Requires pre-aligned face images or will attempt to use face-alignment library
"""

import modal
from pathlib import Path
import io
import base64
from typing import Optional
from fastapi import UploadFile, File

# Define the Modal app
app = modal.App("hairfast-api-simple")

# Create the container image with all dependencies
# Use NVIDIA CUDA base image which includes CUDA toolkit
image = (
    modal.Image.from_registry("nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install(
        "git", "git-lfs", "libglib2.0-0", "libsm6", "libxext6", 
        "libxrender-dev", "libgomp1",
        # OpenCV dependencies
        "libgl1-mesa-glx",
        # Build tool needed for fpie
        "cmake",
    )
    .pip_install(
        "torch==1.13.1",
        "torchvision==0.14.1",
        "ninja==1.11.1",
        "Pillow==10.0.0",
        "numpy==1.23.5",
        "scipy==1.10.1",
        "opencv-python==4.8.0.74",
        "scikit-image==0.21.0",
        "tqdm==4.65.0",
        "matplotlib==3.7.2",
        "gdown==4.7.1",
        "fastapi[standard]==0.115.4",
        # Additional HairFastGAN dependencies
        "face-alignment",
        "dill==0.2.7.1",
        "addict",
        "fpie",
        "dlib-bin",  # Pre-compiled dlib wheel - no building required!
    )
    .pip_install("git+https://github.com/openai/CLIP.git")
    .run_commands(
        # Clone the repository
        "cd /root && git clone https://github.com/AIRI-Institute/HairFastGAN.git",
        # Download pretrained models
        "cd /root && git clone https://huggingface.co/AIRI-Institute/HairFastGAN hairfast_models",
        "cd /root/hairfast_models && git lfs pull",
        # List what we got to verify
        "ls -la /root/hairfast_models/pretrained_models/",
        # Move pretrained models to the HairFastGAN directory
        "cp -r /root/hairfast_models/pretrained_models /root/HairFastGAN/",
        # Also move to root as fallback (some scripts might expect it there)
        "cp -r /root/hairfast_models/pretrained_models /root/",
        # Clean up
        "rm -rf /root/hairfast_models",
        # Verify the models are in place
        "ls -la /root/HairFastGAN/pretrained_models/ || echo 'Warning: pretrained_models not found'",
    )
)

@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    allow_concurrent_inputs=10,
)
class HairFastModel:
    """
    HairFastGAN model wrapper for Modal deployment
    """
    
    @modal.enter()
    def load_model(self):
        """Initialize the model when container starts"""
        import sys
        import os
        import torch
        
        # Change to HairFastGAN directory so relative paths work
        os.chdir('/root/HairFastGAN')
        
        # Set CUDA_HOME for JIT compilation of CUDA extensions
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        
        sys.path.insert(0, '/root/HairFastGAN')
        
        # Check GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! HairFastGAN requires GPU to run.")
        
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Verify pretrained models exist
        model_path = '/root/HairFastGAN/pretrained_models/StyleGAN/ffhq.pt'
        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at {model_path}")
            print("Checking alternate locations...")
            if os.path.exists('/root/pretrained_models/StyleGAN/ffhq.pt'):
                print("Found at /root/pretrained_models/")
            else:
                print("Available files in /root/HairFastGAN/:")
                os.system("find /root/HairFastGAN -name '*.pt' || echo 'No .pt files found'")
        
        from hair_swap import HairFast, get_parser
        
        print("Loading HairFastGAN model...")
        args = get_parser().parse_args([])
        self.hair_fast = HairFast(args)
        print("Model loaded successfully on GPU!")
    
    @modal.method()
    def inference(
        self,
        face_image_bytes: bytes,
        shape_image_bytes: bytes,
        color_image_bytes: Optional[bytes] = None,
        use_alignment: bool = True,
    ) -> bytes:
        """
        Perform hair transfer
        
        Args:
            face_image_bytes: The face image to modify (as bytes)
            shape_image_bytes: The reference image for hair shape (as bytes)
            color_image_bytes: The reference image for hair color (as bytes, optional)
            use_alignment: Whether to use dlib face alignment (default: True for better quality)
        
        Returns:
            Result image as bytes
        """
        from PIL import Image
        import io
        import torch
        import torchvision.transforms as T
        import numpy as np
        
        def preprocess_for_hairfast(img):
            """
            Preprocess image to match HuggingFace demo expectations:
            1. If image is smaller than 1024, resize up first
            2. Center crop to 1024x1024
            """
            # If image is smaller than 1024, resize it proportionally first
            min_dim = min(img.width, img.height)
            if min_dim < 1024:
                scale = 1024 / min_dim
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                print(f"  Upscaled to {img.size} to meet minimum size")
            
            # Now center crop to 1024x1024 (HuggingFace approach)
            square_size = 1024
            left = (img.width - square_size) / 2
            top = (img.height - square_size) / 2
            right = (img.width + square_size) / 2
            bottom = (img.height + square_size) / 2
            
            img_cropped = img.crop((left, top, right, bottom))
            return img_cropped
        
        # Load images from bytes
        face_img = Image.open(io.BytesIO(face_image_bytes)).convert('RGB')
        shape_img = Image.open(io.BytesIO(shape_image_bytes)).convert('RGB')
        
        # If no color image provided, use shape image for color too
        if color_image_bytes:
            color_img = Image.open(io.BytesIO(color_image_bytes)).convert('RGB')
        else:
            color_img = shape_img
        
        print(f"Original images: face={face_img.size}, shape={shape_img.size}, color={color_img.size}")
        
        # Preprocess images
        face_img = preprocess_for_hairfast(face_img)
        shape_img = preprocess_for_hairfast(shape_img)
        color_img = preprocess_for_hairfast(color_img)
        
        print(f"Preprocessed to: {face_img.size}")
        
        # Optionally use dlib face alignment for even better results
        if use_alignment:
            try:
                import sys
                sys.path.insert(0, '/root/HairFastGAN')
                from utils.shape_predictor import align_face
                
                print("Applying face alignment...")
                face_img = Image.fromarray(align_face(np.array(face_img)))
                shape_img = Image.fromarray(align_face(np.array(shape_img)))
                color_img = Image.fromarray(align_face(np.array(color_img)))
                print(f"Face-aligned to: {face_img.size}")
            except Exception as e:
                print(f"Face alignment skipped (not critical): {e}")
        
        # Perform hair transfer using swap method (exactly like HuggingFace)
        result = self.hair_fast.swap(face_img, shape_img, color_img)
        
        # Convert result tensor to PIL Image (exactly like HuggingFace)
        result_pil = T.functional.to_pil_image(result)
        
        # Convert result to bytes
        output_buffer = io.BytesIO()
        result_pil.save(output_buffer, format='PNG')
        output_bytes = output_buffer.getvalue()
        
        print("Hair transfer completed successfully!")
        return output_bytes


@app.function(image=image)
@modal.web_endpoint(method="POST", docs=True)
async def transfer_hair(
    face_image: str,
    shape_image: str,
    color_image: Optional[str] = None,
) -> dict:
    """
    Web API endpoint for hair transfer
    
    Expects base64-encoded images in the request body:
    {
        "face_image": "base64_encoded_image",
        "shape_image": "base64_encoded_image",
        "color_image": "base64_encoded_image"  // optional
    }
    
    Returns:
    {
        "success": true,
        "result_image": "base64_encoded_result"
    }
    """
    try:
        # Decode base64 images
        face_bytes = base64.b64decode(face_image)
        shape_bytes = base64.b64decode(shape_image)
        color_bytes = base64.b64decode(color_image) if color_image else None
        
        # Call the model
        model = HairFastModel()
        result_bytes = model.inference.remote(face_bytes, shape_bytes, color_bytes)
        
        # Encode result as base64
        result_base64 = base64.b64encode(result_bytes).decode('utf-8')
        
        return {
            "success": True,
            "result_image": result_base64
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.function(image=image)
@modal.web_endpoint(method="POST", docs=True)
async def transfer_hair_multipart(
    face_image: UploadFile = File(...),
    shape_image: UploadFile = File(...),
    color_image: Optional[UploadFile] = File(None)
) -> dict:
    """
    Alternative endpoint that accepts multipart/form-data file uploads
    
    Upload files with keys:
    - face_image: The face image file
    - shape_image: The hair shape reference image file
    - color_image: The hair color reference image file (optional)
    
    Returns:
    {
        "success": true,
        "result_image": "base64_encoded_result"
    }
    """
    try:
        # Read file contents
        face_bytes = await face_image.read()
        shape_bytes = await shape_image.read()
        color_bytes = await color_image.read() if color_image else None
        
        # Call the model
        model = HairFastModel()
        result_bytes = model.inference.remote(face_bytes, shape_bytes, color_bytes)
        
        # Encode result as base64
        result_base64 = base64.b64encode(result_bytes).decode('utf-8')
        
        return {
            "success": True,
            "result_image": result_base64
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }