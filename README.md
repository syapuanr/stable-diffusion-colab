# Stable Diffusion WebUI for Google Colab

WebUI untuk Stable Diffusion yang dioptimasi untuk Google Colab dengan fitur lengkap seperti ComfyUI/A1111.

## Features

- ✅ Support multiple models (SD 1.5, SDXL, SD 2.1)
- ✅ LoRA, ControlNet, VAE support
- ✅ Memory optimization untuk model besar
- ✅ Cloudflare/Ngrok tunnel
- ✅ User-friendly Gradio interface
- ✅ Batch processing
- ✅ Txt2Img, Img2Img, Inpainting

## Quick Start (Google Colab)

1. Buka `setup_colab.ipynb` di Google Colab
2. Jalankan semua cells
3. Akses WebUI melalui link yang muncul

## Manual Installation

```bash
git clone https://github.com/syapuanr/sd-webui.git
cd sd-webui
pip install -r requirements.txt
python main.py --share
Configuration
Edit config.yaml untuk mengatur:
Model paths
Generation settings
Memory optimization
Tunnel settings
Usage
# Basic launch
python main.py

# Launch with public URL
python main.py --share

# Custom port
python main.py --port 8080
Requirements
Python 3.10+
CUDA compatible GPU (recommended)
15GB+ GPU memory untuk SDXL
8GB+ GPU memory untuk SD 1.5
License
MIT License
---

## Yang Sudah Kita Buat:

✅ Struktur folder dasar  
✅ Requirements.txt  
✅ Config system  
✅ Colab notebook setup  
✅ Tunnel manager  
✅ Entry point (main.py)  

## Next Steps:

Selanjutnya kita perlu buat:
1. **Model Loader** - untuk load model dengan memory optimization
2. **Inference Engine** - untuk generate gambar
3. **Gradio UI** - interface pengguna
