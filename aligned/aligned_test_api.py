"""
Quick test script for your deployed HairFastGAN API
Saves results with timestamps to avoid overwriting previous tests.
"""

import requests
import base64
import os
from pathlib import Path
from datetime import datetime

# 1. Update this to your specific Modal URL
API_URL_MULTIPART = "https://sebasdeldi123--hairfast-api-exact-transfer-hair.modal.run"

def test_multipart_upload(face_path, shape_path, color_path=None):
    """Test the multipart file upload endpoint and save with timestamp"""
    print(f"\n🧪 Testing HairFastGAN Transfer...")
    
    # 2. Generate a unique timestamp (Format: YYYYMMDD_HHMMSS)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract filenames for the output label
    face_name = Path(face_path).stem
    shape_name = Path(shape_path).stem
    
    # 3. Create the unique output filename
    output_filename = f"result_{timestamp}_{face_name}_to_{shape_name}.png"
    
    # Verify local files exist
    for p in [face_path, shape_path]:
        if not os.path.exists(p):
            print(f"   ✗ Local Error: File not found at {p}")
            return False
    
    try:
        files = {
            'face_image': open(face_path, 'rb'),
            'shape_image': open(shape_path, 'rb'),
        }
        
        if color_path and os.path.exists(color_path):
            files['color_image'] = open(color_path, 'rb')
        
        print(f"   🛰  Sending request to Modal...")
        response = requests.post(API_URL_MULTIPART, files=files, timeout=120)
        
        # Close file handles
        for f in files.values():
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # Decode the base64 result
                result_bytes = base64.b64decode(result["result_image"])
                
                # Write the uniquely named file
                with open(output_filename, 'wb') as f:
                    f.write(result_bytes)
                
                print(f"   ✓ Success!")
                print(f"   📂 Saved as: {output_filename}")
                return True
            else:
                print(f"   ✗ API Error: {result.get('error')}")
                return False
        else:
            print(f"   ✗ HTTP Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print(f"HairFastGAN API Test Client | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Run the test
    test_multipart_upload(
        face_path="test_images/1.jpg",
        shape_path="test_images/3.jpg",
        color_path="test_images/4.jpg"
    )
