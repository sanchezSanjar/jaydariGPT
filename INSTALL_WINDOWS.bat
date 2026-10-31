@echo off
cd /d "%~dp0"
set CUDA_INDEX=https://download.pytorch.org/whl/cu132

if exist ".venv\Scripts\python.exe" (
  echo Using the existing .venv
) else (
  py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
  if errorlevel 1 (
    echo Could not create a Python environment. Install Python 3.11+ 64-bit and retry.
    pause
    exit /b 1
  )
)
set PY=.venv\Scripts\python.exe
"%PY%" -m pip install --upgrade pip

echo.
where nvidia-smi >nul 2>nul && nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo No NVIDIA GPU found. Installing CPU PyTorch.
  "%PY%" -m pip install torch
) else (
  echo NVIDIA GPU found. Installing GPU ^(CUDA^) PyTorch - this is a large download.
  "%PY%" -m pip install torch --index-url %CUDA_INDEX%
  if errorlevel 1 (
    echo GPU build failed to install. Falling back to CPU PyTorch.
    "%PY%" -m pip install torch
  )
)
"%PY%" -m pip install -r requirements.txt

echo.
"%PY%" -c "import torch; print('PyTorch', torch.__version__, '- device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo.
echo Dependencies installed. Next, while connected to internet, run:
echo     .venv\Scripts\python.exe download_base_model.py
echo Then double-click JaydariGPT.exe
pause
