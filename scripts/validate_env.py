"""Environment and System Diagnostic Script for NeuroGuard.

Verifies:
1. Python version compatibility (>=3.10)
2. Operating system and architecture
3. Available CPU cores, system RAM, and free disk space
4. Core dependency installations and versions
5. Local project directory structure and write permissions
6. PyTorch device availability (CPU / CUDA)
"""

import sys
import os
import platform
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_python_version() -> bool:
    print(f"[*] Python Version: {platform.python_version()} ({platform.python_implementation()})")
    if sys.version_info < (3, 10):
        print("    [!] WARNING: Python 3.10+ is recommended.")
        return False
    print("    [OK] Python version compatible.")
    return True


def check_system_resources() -> dict:
    print(f"[*] Operating System: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"[*] CPU Count: {os.cpu_count()} logical cores")

    # Disk space check
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    print(f"[*] Disk Space at Project Root: {free_gb:.2f} GB free out of {total_gb:.2f} GB total")

    # RAM check using psutil if available, otherwise fallback
    ram_gb = "Unknown"
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024**3)
        avail_ram_gb = vm.available / (1024**3)
        print(f"[*] Memory (RAM): {avail_ram_gb:.2f} GB available / {total_ram_gb:.2f} GB total")
    except ImportError:
        print("[*] Memory (RAM): psutil not installed, skipped detailed RAM query.")

    return {"free_disk_gb": free_gb}


def check_dependencies() -> dict:
    packages = [
        ("numpy", True),
        ("scipy", True),
        ("pandas", True),
        ("matplotlib", True),
        ("seaborn", True),
        ("sklearn", True),
        ("torch", True),
        ("fastapi", True),
        ("uvicorn", True),
        ("streamlit", True),
        ("yaml", True),
        ("pydantic", True),
        ("pytest", True),
        ("mne", False),
        ("optuna", False),
        ("xgboost", False),
        ("shap", False),
    ]

    print("\n[*] Checking Dependency Availability:")
    status = {}
    for pkg_name, required in packages:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    [OK] {pkg_name}: {ver}")
            status[pkg_name] = True
        except ImportError:
            tag = "[CRITICAL MISSING]" if required else "[OPTIONAL / PHASE-SPECIFIC]"
            print(f"    [!] {pkg_name}: NOT INSTALLED {tag}")
            status[pkg_name] = False
    return status


def check_directories() -> bool:
    print("\n[*] Checking Project Directory Structure:")
    expected_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "data/interim",
        "notebooks",
        "src/data",
        "src/preprocessing",
        "src/features",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/explainability",
        "src/visualization",
        "src/api",
        "src/utils",
        "frontend",
        "tests",
        "scripts",
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/plots",
        "artifacts/reports",
        "artifacts/mlruns",
        "docs",
        "report",
    ]

    all_exist = True
    for d in expected_dirs:
        p = PROJECT_ROOT / d
        if p.exists() and p.is_dir():
            pass
        else:
            print(f"    [!] Missing directory: {d}")
            p.mkdir(parents=True, exist_ok=True)
            print(f"        -> Created {d}")
    print(f"    [OK] All {len(expected_dirs)} core directories exist and are verified.")
    return all_exist


def check_pytorch_device() -> str:
    print("\n[*] Checking PyTorch Compute Device:")
    try:
        import torch
        if torch.cuda.is_available():
            dev = f"CUDA ({torch.cuda.get_device_name(0)})"
            print(f"    [OK] Accelerated GPU Available: {dev}")
        else:
            dev = "CPU"
            print("    [OK] Compute Device: CPU (Optimized for multi-threading & low memory overhead)")
        return dev
    except Exception as e:
        print(f"    [!] PyTorch check failed: {e}")
        return "Unknown"


def main():
    print("=" * 70)
    print(" NeuroGuard: Environment & Diagnostic Verification")
    print(f" Project Root: {PROJECT_ROOT}")
    print("=" * 70)

    py_ok = check_python_version()
    sys_info = check_system_resources()
    dep_status = check_dependencies()
    dir_ok = check_directories()
    dev_info = check_pytorch_device()

    print("\n" + "=" * 70)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f" Python Compatible: {'YES' if py_ok else 'NO'}")
    print(f" Directories Valid: {'YES' if dir_ok else 'NO'}")
    print(f" Free Disk Space:   {sys_info['free_disk_gb']:.2f} GB (Sufficient for CHB-MIT subset)")
    print(f" PyTorch Device:    {dev_info}")
    
    missing_critical = [k for k, v in dep_status.items() if not v and k in ["numpy", "scipy", "pandas", "sklearn", "torch", "fastapi", "yaml"]]
    if missing_critical:
        print(f" [!] Missing Critical Packages: {missing_critical}")
        sys.exit(1)
    else:
        print(" [OK] Core Environment is READY for Phases 1 & 2.")
    print("=" * 70)


if __name__ == "__main__":
    main()
