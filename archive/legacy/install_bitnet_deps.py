#!/usr/bin/env python3
"""
BitNet Dependencies Installation Script

This script ensures that all BitNet dependencies are properly installed
across different platforms (NVIDIA CUDA, Apple Silicon, CPU-only).
"""

import os
import sys
import subprocess
import platform
import importlib.util
from pathlib import Path

def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name, extra_args=None):
    """Install a package using pip."""
    cmd = [sys.executable, "-m", "pip", "install", package_name]
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        subprocess.check_call(cmd)
        print(f"✅ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def detect_platform():
    """Detect the current platform and capabilities."""
    system = platform.system()
    machine = platform.machine()
    
    platform_info = {
        "system": system,
        "machine": machine,
        "is_apple_silicon": system == "Darwin" and machine == "arm64",
        "is_linux_x64": system == "Linux" and machine == "x86_64",
        "is_windows": system == "Windows",
        "has_cuda": False,
        "has_mps": False,
    }
    
    # Check for CUDA availability
    try:
        import torch
        platform_info["has_cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check for MPS availability (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps'):
            platform_info["has_mps"] = torch.backends.mps.is_available()
    except ImportError:
        pass
    
    return platform_info

def install_bitnet_dependencies():
    """Install BitNet-specific dependencies based on platform."""
    
    print("🔧 BitNet Dependencies Installation")
    print("=" * 50)
    
    platform_info = detect_platform()
    print(f"Platform: {platform_info['system']} {platform_info['machine']}")
    
    success = True
    required_packages = []
    
    # Core BitNet dependencies
    core_deps = [
        "torch>=2.0.0",
        "einops>=0.8.0",
        "zetascale>=2.1.6",
        "numpy>=1.21.0",
        "psutil>=5.8.0",
    ]
    
    # Platform-specific packages
    if platform_info["is_apple_silicon"]:
        print("🍎 Detected Apple Silicon - installing MPS-optimized packages")
        optional_deps = [
            "torch-metal",  # For Apple Silicon GPU acceleration
        ]
    elif platform_info["has_cuda"]:
        print("🚀 Detected CUDA - installing CUDA-optimized packages")
        optional_deps = [
            "cupy-cuda12x",  # For CUDA acceleration (if available)
        ]
    else:
        print("💻 Detected CPU-only system")
        optional_deps = []
    
    # Install core dependencies
    print(f"\n📦 Installing core BitNet dependencies...")
    for package in core_deps:
        if not install_package(package):
            success = False
    
    # Install optional dependencies
    if optional_deps:
        print(f"\n🔧 Installing platform-specific optimizations...")
        for package in optional_deps:
            install_package(package)  # Don't fail if these don't install
    
    return success

def verify_bitnet_installation():
    """Verify that BitNet can be imported and used."""
    
    print(f"\n🧪 Verifying BitNet Installation")
    print("-" * 40)
    
    # Test core dependencies
    test_packages = [
        ("torch", "PyTorch"),
        ("einops", "Einops"),
        ("zetascale", "ZetaScale"),
        ("numpy", "NumPy"),
        ("psutil", "psutil"),
    ]
    
    all_good = True
    
    for package, name in test_packages:
        if check_package_installed(package):
            print(f"✅ {name} - Available")
        else:
            print(f"❌ {name} - Missing")
            all_good = False
    
    # Test BitNet compatible module
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from bit_vla.models.bitnet_compatible import BitLinearCompatible
        print("✅ BitNet Compatible Module - Available")
    except ImportError as e:
        print(f"❌ BitNet Compatible Module - Import Error: {e}")
        all_good = False
    
    # Test platform-specific features
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA Support - Available")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS Support - Available")
        else:
            print("ℹ️ CPU-only mode")
    except ImportError:
        pass
    
    return all_good

def create_platform_test():
    """Create a simple test to verify BitNet works on this platform."""
    
    test_script = """
# BitNet Platform Test
import torch
import sys
import traceback

def test_bitnet():
    try:
        # Test basic imports
        sys.path.insert(0, "src")
        from bit_vla.models.bitnet_compatible import BitLinearCompatible
        
        # Create a simple BitNet layer
        layer = BitLinearCompatible(32, 16)
        
        # Test forward pass
        x = torch.randn(1, 32)
        output = layer(x)
        
        print(f"✅ BitNet test successful!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ BitNet test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bitnet()
"""
    
    with open("test_bitnet_installation.py", "w") as f:
        f.write(test_script)
    
    print(f"\n📝 Created test script: test_bitnet_installation.py")
    print(f"   Run: python test_bitnet_installation.py")

def main():
    """Main installation function."""
    
    print("🚀 BitNet Cross-Platform Installation")
    print("=" * 60)
    
    # Install dependencies
    if install_bitnet_dependencies():
        print(f"\n✅ BitNet dependencies installed successfully!")
    else:
        print(f"\n❌ Some dependencies failed to install")
        return False
    
    # Verify installation
    if verify_bitnet_installation():
        print(f"\n🎉 BitNet installation verified!")
    else:
        print(f"\n⚠️ BitNet installation has issues")
    
    # Create test script
    create_platform_test()
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Run: python test_bitnet_installation.py")
    print(f"   2. If successful, run: python cross_platform_bitnet_evaluation.py --mode benchmark")
    print(f"   3. For full installation: pip install -e .")

if __name__ == "__main__":
    main() 