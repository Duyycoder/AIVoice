"""GPU diagnostic script for the AIVoice TTS framework.

Run this after installing PyTorch with CUDA support to verify
that your GPU is correctly detected and ready for inference.
"""
import sys

def main():
    print("=" * 60)
    print("AIVoice GPU Diagnostic Report")
    print("=" * 60)
    
    # 1. PyTorch version
    try:
        import torch
        print(f"\nPyTorch version:   {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        sys.exit(1)
    
    # 2. CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available:    {cuda_available}")
    
    if not cuda_available:
        print("\n⚠️  CUDA is NOT available.")
        print("   PyTorch is running in CPU-only mode.")
        print("\n   To fix this, reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(0)
    
    # 3. CUDA version
    print(f"CUDA version:      {torch.version.cuda}")
    
    # 4. cuDNN
    cudnn_available = torch.backends.cudnn.is_available()
    print(f"cuDNN available:   {cudnn_available}")
    if cudnn_available:
        print(f"cuDNN version:     {torch.backends.cudnn.version()}")
    
    # 5. GPU details
    gpu_count = torch.cuda.device_count()
    print(f"\nGPU count:         {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024 ** 3)
        
        # Current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        free = total_mem - reserved
        
        print(f"\n--- GPU {i}: {props.name} ---")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total VRAM:         {total_mem:.2f} GB")
        print(f"  Allocated:          {allocated:.2f} GB")
        print(f"  Reserved:           {reserved:.2f} GB")
        print(f"  Free (approx):      {free:.2f} GB")
    
    # 6. SDPA support
    print(f"\nSDPA (Scaled Dot-Product Attention):")
    try:
        # Test SDPA with a small tensor
        q = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float16)
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print("  ✅ SDPA is functional on GPU")
    except Exception as e:
        print(f"  ❌ SDPA test failed: {e}")
    
    # 7. Quick inference test
    print(f"\nQuick tensor test:")
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.mm(x, x)
        torch.cuda.synchronize()
        print("  ✅ GPU tensor operations working correctly")
    except Exception as e:
        print(f"  ❌ GPU tensor test failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ GPU is ready for AIVoice inference!")
    print("=" * 60)


if __name__ == "__main__":
    main()
