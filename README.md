
# HairFastGAN Standalone Modal API

This project provides a production-ready, high-performance API for **HairFastGAN**, deployed on [Modal](https://modal.com). It is designed to be **fully self-contained**, ensuring faster cold starts and 100% uptime by baking model weights directly into the container.

## 📂 Project Structure

```text
HairStyle/
├── aligned/
│   ├── hairfast_aligned.py    # Modal deployment script
│   └── aligned_test_api.py    # Local test client script
├── pretrained_models/         # Local model weights (Baked into image)
│   ├── StyleGAN/              # ffhq.pt (~127MB)
│   ├── Blending/              
│   └── Segmentation/          
├── test_images/               # Folder containing sample images for testing
└── README.md

```

---

## 🚀 Setup & Deployment Steps

### 1. Prerequisite Installations

You need the following tools installed on your local machine:

* **Modal CLI:** `pip install modal`
* **Git LFS:** Required to download large model binaries.
* *macOS:* `brew install git-lfs`
* *Linux:* `sudo apt-get install git-lfs`


* **Hugging Face Account:** You need an [Access Token](https://huggingface.co/settings/tokens) (Read access).

### 2. Prepare the Model Weights

Because this project is "standalone," you must download the weights to your machine once before deploying.

1. **Clone and Pull Binaries:**
```bash
git lfs install
git clone https://huggingface.co/AIRI-Institute/HairFastGAN temp_hf_models
cd temp_hf_models && git lfs pull && cd ..

```


2. **Isolate for Project:**
Move the weights into your project root and clean up:
```bash
mkdir -p ./pretrained_models
cp -R ./temp_hf_models/pretrained_models/ ./pretrained_models/
rm -rf temp_hf_models  # Cleanup temporary folder

```



### 3. Deploy to Modal

Navigate to your project root and run:

```bash
modal deploy aligned/hairfast_aligned.py

```

---

## 🧪 Testing the API

A test client is provided in `aligned/aligned_test_api.py` to verify your deployment using local images.

### 1. Configure the Client

Open `aligned/aligned_test_api.py` and update the `API_URL_MULTIPART` with the URL provided by Modal after deployment:

```python
API_URL_MULTIPART = "https://your-username--hairfast-api-standalone-transfer-hair.modal.run"

```

### 2. Run the Test

Ensure you have test images in a `test_images/` folder, then run:

```bash
python aligned/aligned_test_api.py

```

### 3. Verify Results

If successful, the script will output:
`✓ Success! Result saved to result_multipart_test.png`

---

## 🛠 API Documentation

### `POST /transfer_hair`

Accepts `multipart/form-data` containing images.

**Parameters:**
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `face_image` | File | Yes | The source face image. |
| `shape_image` | File | Yes | The target hairstyle reference image. |
| `color_image` | File | No | Optional reference for hair color. |

**Response Format:**

```json
{
  "success": true,
  "result_image": "iVBORw0KGgoAAAANSUh..." // Base64 PNG string
}

```

---

## 🧠 Core Technologies & Credits

* **Model:** [AIRI Institute / HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN)
* **Poisson Blending:** [fpie (Fast Poisson Image Editing)](https://www.google.com/search?q=https://github.com/li-plus/fpie)
* **Cloud Infrastructure:** [Modal](https://modal.com)

---
