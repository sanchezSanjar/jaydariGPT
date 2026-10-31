@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 (
  echo Could not create Python 3.11 environment. Install Python 3.11 and retry.
  pause
  exit /b 1
)
call .venv\\Scripts\\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing CPU PyTorch. For NVIDIA GPU, install the matching CUDA build from pytorch.org first.
python -m pip install torch
python -m pip install -r requirements.txt
echo.
echo Dependencies installed. Next, while connected to internet, run:
echo     .venv\Scripts\python.exe download_base_model.py
echo Then disconnect internet and double-click START_JAYDARIGPT.bat
pause
