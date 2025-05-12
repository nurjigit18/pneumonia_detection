import torch


# Comprehensive CUDA Availability Troubleshooting Guide

def check_cuda_installation():
    """
    Comprehensive checks for CUDA and PyTorch GPU availability
    """
    import sys
    import torch
    import subprocess

    print("CUDA Availability Troubleshooting Checklist:")
    
    # 1. Check PyTorch CUDA Availability
    print(f"\n1. PyTorch CUDA Availability:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    # 2. Check CUDA Version
    try:
        cuda_version = torch.version.cuda
        print(f"\n2. CUDA Version in PyTorch: {cuda_version}")
    except Exception as e:
        print(f"\n2. Could not retrieve CUDA version: {e}")
    
    # 3. Check GPU Information
    try:
        print("\n3. GPU Information:")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error retrieving GPU information: {e}")
    
    # 4. System CUDA Check
    print("\n4. System CUDA Checks:")
    try:
        # Check NVIDIA-SMI
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode('utf-8')
        print("NVIDIA-SMI is available and functioning.")
        print(nvidia_smi_output)
    except FileNotFoundError:
        print("NVIDIA-SMI not found. CUDA might not be properly installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
    
    # 5. Potential Solutions
    print("\n5. Potential Solutions:")
    print("If CUDA is not available, try the following:")
    solutions = [
        "1. Verify NVIDIA GPU drivers are up to date",
        "2. Reinstall CUDA toolkit matching your PyTorch version",
        "3. Reinstall PyTorch with CUDA support",
        "4. Ensure CUDA is in your system PATH",
        "5. Check compatibility between PyTorch, CUDA, and your GPU"
    ]
    for solution in solutions:
        print(solution)

# Example usage of CUDA if available
def cuda_device_example():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nCUDA is available!")
        print(f"Using device: {device}")
        
        # Move a tensor to GPU
        x = torch.rand(5, 3)
        x_gpu = x.to(device)
        print("Tensor moved to GPU:", x_gpu)
    else:
        print("\nCUDA is not available. Using CPU.")
        device = torch.device("cpu")

# Run the checks
if __name__ == "__main__":
    check_cuda_installation()
    cuda_device_example()