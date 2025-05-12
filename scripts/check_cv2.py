# find_missing_dll.py
import os
import sys
import ctypes
from ctypes import wintypes

# Define the necessary Windows API functions
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
SetErrorMode = kernel32.SetErrorMode
SetErrorMode.argtypes = [wintypes.UINT]
SetErrorMode.restype = wintypes.UINT

# Constants for SetErrorMode
SEM_FAILCRITICALERRORS = 0x0001
SEM_NOGPFAULTERRORBOX = 0x0002
SEM_NOOPENFILEERRORBOX = 0x8000

# Function to log DLL loading issues
def log_dll_load(dll_path):
    try:
        # Suppress error dialogs
        old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX)
        
        # Try to load the DLL
        lib = ctypes.CDLL(dll_path)
        SetErrorMode(old_mode)
        return True, None
    except Exception as e:
        SetErrorMode(old_mode)
        return False, str(e)

# Paths to check
site_packages = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages')
cv2_paths = [
    os.path.join(site_packages, 'cv2'),
    os.path.join(site_packages, 'opencv_python.libs'),
    os.path.join(os.path.dirname(sys.executable), 'DLLs')
]

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Checking DLLs in these paths:")

# Check all relevant directories for DLLs
for path in cv2_paths:
    print(f"\nChecking {path}")
    if os.path.exists(path):
        dlls = [f for f in os.listdir(path) if f.endswith('.dll') or f.endswith('.pyd')]
        
        if not dlls:
            print("  No DLL/PYD files found")
        else:
            print(f"  Found {len(dlls)} DLL/PYD files")
            for dll in dlls:
                dll_path = os.path.join(path, dll)
                success, error = log_dll_load(dll_path)
                if success:
                    print(f"  ✓ {dll}: Loaded successfully")
                else:
                    print(f"  ✗ {dll}: Failed to load - {error}")
    else:
        print("  Directory does not exist")

# Try to identify the specific issue with cv2 module
try:
    print("\nAttempting to import cv2 with detailed error tracking...")
    import importlib.util
    import traceback
    
    try:
        spec = importlib.util.find_spec('cv2')
        if spec:
            print(f"cv2 spec found at: {spec.origin}")
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print("cv2 loaded successfully!")
            except Exception as e:
                print(f"Error during module loading: {e}")
                traceback.print_exc()
        else:
            print("cv2 spec not found")
    except Exception as e:
        print(f"Error finding spec: {e}")
        traceback.print_exc()
    
except Exception as e:
    print(f"General error: {e}")
    traceback.print_exc()