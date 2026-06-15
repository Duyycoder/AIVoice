import os
import sys
import argparse
import requests
from tqdm import tqdm

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1 MB blocks
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        print("Download finished successfully.")
    except Exception as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        if os.path.exists(output_path):
            os.remove(output_path)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download model weights for Piper and/or XTTSv2 clone engines.")
    parser.add_argument(
        "--engine",
        choices=["piper", "clone", "all"],
        default="piper",
        help="Specify which model files to download: 'piper', 'clone', or 'all' (default: piper)."
    )
    args = parser.parse_args()
    
    models_dir = "models"
    
    # 1. Piper model files
    if args.engine in ["piper", "all"]:
        print("Checking Piper model files...")
        piper_base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium"
        piper_files = ["vi_VN-vais1000-medium.onnx", "vi_VN-vais1000-medium.onnx.json"]
        
        for filename in piper_files:
            url = f"{piper_base_url}/{filename}"
            out_path = os.path.join(models_dir, "piper", filename)
            if not os.path.exists(out_path):
                download_file(url, out_path)
            else:
                print(f"{filename} already exists at {out_path}.")
                
    # 2. XTTSv2 model files
    if args.engine in ["clone", "all"]:
        print("Checking XTTSv2 model files...")
        xtts_dir = os.path.join(models_dir, "xtts_v2")
        
        # Files to download from thivux/XTTS-v2-vietnamse
        thivux_base_url = "https://huggingface.co/thivux/XTTS-v2-vietnamse/resolve/main"
        thivux_files = {
            "config.json": "config.json",
            "vocab.json": "vocab.json",
            "best_model.pth": "model.pth"
        }
        
        # Files to download from coqui/XTTS-v2 (base model assets required by fine-tuned model)
        coqui_base_url = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
        coqui_files = {
            "dvae.pth": "dvae.pth",
            "mel_stats.pth": "mel_stats.pth",
            "speakers_xtts.pth": "speakers_xtts.pth"
        }
        
        # Download thivux files
        for src_name, dest_name in thivux_files.items():
            url = f"{thivux_base_url}/{src_name}"
            out_path = os.path.join(xtts_dir, dest_name)
            if not os.path.exists(out_path):
                download_file(url, out_path)
            else:
                print(f"{dest_name} already exists at {out_path}.")
                
        # Download coqui base files
        for src_name, dest_name in coqui_files.items():
            url = f"{coqui_base_url}/{src_name}"
            out_path = os.path.join(xtts_dir, dest_name)
            if not os.path.exists(out_path):
                download_file(url, out_path)
            else:
                print(f"{dest_name} already exists at {out_path}.")

if __name__ == "__main__":
    main()
