# Environment

Local machine specs for this workspace root (`D:\Master\hk3\msc`). Sub-projects can assume this environment unless they state otherwise.

- OS: Windows 10 Pro, Build 19041 (x64)
- CPU: Intel Core i5-12600K (10 cores / 16 logical processors)
- RAM: 16 GB (16,158 MB)
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM (driver 536.23, supports CUDA 12.2); integrated Intel UHD Graphics 770 also present
- Python: 3.10.6 (`python` on PATH; `python3` not aliased)
- PyTorch: 2.6.0+cu124 installed, `torch.cuda.is_available() == True` — GPU-accelerated training/inference works out of the box
- CUDA toolkit (`nvcc`, the compiler): not installed — irrelevant for pip-installed PyTorch, which bundles its own CUDA runtime; only matters if compiling custom CUDA kernels from source
- Shell: PowerShell (primary)
- C++ compiler: `g++` available (MSYS2 build, g++ 15.2.0) — usable for compiling/running C++ to get objective ground truth (e.g. compile success, test-case pass/fail), not just LLM self-assessment


# Run background
- Show the user command to get log for monitoring
