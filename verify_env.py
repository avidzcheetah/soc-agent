import sys

def main():
    print("===================================")
    print("Environment Verification")
    print("===================================")
    
    # Python Version
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Try loading Torch
    try:
        import torch
        print(f"Torch {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available {'✓' if cuda_available else '✗'}")
        
        if cuda_available:
            print(f"GPU\n{torch.cuda.get_device_name(0)}")
            
            # Convert bytes to GB and round
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3))
            print(f"VRAM\n{vram_gb}GB")
        else:
            print("GPU\nNone")
            print("VRAM\n0GB")
            
    except ImportError:
        print("Torch Not Installed")
        cuda_available = False
        
    # Try loading Transformers
    try:
        import transformers
        print(f"Transformers\n{transformers.__version__}")
    except ImportError:
        print("Transformers\nNot Installed")
        
    print("===================================")
    
    if not cuda_available:
        print("\nIf GPU is missing, stop here.")
        sys.exit(1)

if __name__ == "__main__":
    main()
