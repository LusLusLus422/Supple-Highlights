#!/usr/bin/env python3
"""
Setup script for downloading and configuring the LLaMA model.
This script downloads a quantized version of LLaMA 2 7B optimized for Apple Silicon.
"""

import os
import sys
import requests
from pathlib import Path
import hashlib
from tqdm import tqdm

# Model configuration
MODEL_CONFIG = {
    'name': 'llama-2-7b-chat.Q4_K_M.gguf',
    'url': 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf',
    'expected_size': 3791728640,  # ~3.8GB
}

def download_file(url: str, dest_path: Path, expected_size: int) -> bool:
    """Download a file with progress bar."""
    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        print(f"Model already exists at {dest_path}")
        return True
        
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    if total_size != expected_size:
        print(f"Warning: Expected size ({expected_size}) differs from download size ({total_size})")
    
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    try:
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
    except Exception as e:
        print(f"Error downloading file: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False
    finally:
        progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("Error: download incomplete")
        if dest_path.exists():
            dest_path.unlink()
        return False
    
    return True

def main():
    # Create models directory
    models_dir = Path('models/llama')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model
    model_path = models_dir / MODEL_CONFIG['name']
    print(f"Downloading {MODEL_CONFIG['name']}...")
    
    success = download_file(
        MODEL_CONFIG['url'],
        model_path,
        MODEL_CONFIG['expected_size']
    )
    
    if success:
        print(f"\nModel successfully downloaded to {model_path}")
        print("\nYou can now use the model with the following configuration:")
        print("""
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama/llama-2-7b-chat.Q4_K_M.gguf",
    n_gpu_layers=-1,  # Use all available GPU layers
    n_ctx=2048,       # Context window
    n_batch=512       # Batch size for prompt processing
)
        """)
    else:
        print("\nError: Failed to download model")
        sys.exit(1)

if __name__ == '__main__':
    main() 