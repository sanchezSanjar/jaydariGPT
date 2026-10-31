# JaydariGPT — Local Python Edition

This package runs your supplied JaydariGPT LoRA adapter from ordinary Python on Windows, macOS, or Linux. It does not need Google Colab.

## Important: what is and is not included

Included:
- Your trained JaydariGPT LoRA adapter (`models/jaydari_gpt_adapter/`)
- The tokenizer saved with your adapter
- Python inference and training code
- Windows setup and launch scripts

**The original TinyLlama 1.1B base-model weights were NOT present in either uploaded ZIP.** A LoRA adapter is only the learned update; it cannot generate text by itself. The base model is several gigabytes, so it is not possible to reconstruct it from the adapter. This ZIP therefore does not contain the base weights.

You must download the base model once while connected to the internet. After that download completes, chat runs entirely from local files and does not need internet. Python packages must also be installed before going offline.

## Quick start on Windows

Recommended: Python 3.11 (64-bit), with several GB of free disk space. A GPU is optional; CPU inference works but is slower and uses more RAM.

1. Extract this ZIP to a normal folder, for example `C:\AI\JaydariGPT_Local`.
2. Open the extracted folder.
3. Double-click `INSTALL_WINDOWS.bat`. It creates `.venv` and installs CPU PyTorch plus the Python packages. This step needs internet.
4. While still online, open Command Prompt in this folder and run:

   ```bat
   .venv\Scripts\python.exe download_base_model.py
   ```

   This downloads TinyLlama into `models\TinyLlama-1.1B-Chat-v1.0`. It may take a while and needs several GB of free disk space.
5. When the download finishes, disconnect from the internet if you want.
6. Double-click **`JaydariGPT.exe`** to open the chat dashboard. No console window appears.

### JaydariGPT.exe

- It is a small launcher (about 16 KB) that starts the dashboard from this folder's `.venv`. Keep it inside the `JaydariGPT_Local` folder; to open it from the Desktop, right-click it → **Send to → Desktop (create shortcut)**, or pin it to the taskbar.
- If setup is not finished, it tells you which step to run.
- App output and errors go to `jaydarigpt.log` in this folder.
- Windows SmartScreen may warn the first time because the exe is not code-signed: choose **More info → Run anyway**.
- To rebuild it (for example after changing the icon), double-click `launcher\build_launcher.bat`. It uses the C# compiler built into Windows, so nothing extra is needed.
- `START_JAYDARIGPT.bat` still works and shows a console, which is handy for debugging.

The desktop interface uses Python’s built-in Tkinter library, so it does not require a separate GUI package. The model loads in the background while the window remains responsive.

You can also launch the command-line chat from a terminal:

```bat
.venv\Scripts\python.exe main.py chat
```

Type `exit` to stop chatting.

## GPU setup

The included Windows installer installs CPU PyTorch for maximum compatibility. If your PC has an NVIDIA GPU, install a PyTorch CUDA build that matches your driver from https://pytorch.org/get-started/locally/ before running inference. The inference loader automatically uses CUDA when available. The CPU version remains supported.

## Offline guarantee and boundaries

- **Chat/inference:** offline after the base model and Python packages have been installed.
- **Tokenization demo:** uses the local base tokenizer after the base-model download.
- **Training:** not automatically offline. The original training configuration uses the Hugging Face `yahma/alpaca-cleaned` dataset. To train offline, first download that dataset or provide a local dataset and update the dataset path. Training also needs substantial RAM/VRAM and can be very slow on CPU.
- Do not delete the `models` folder; it contains the adapter and, after setup, the base model.

## Commands

```bat
python main.py chat
python main.py chat --max-new-tokens 400
python main.py tokenize
python main.py train --max-samples 100 --epochs 1
python main.py --help
```

The chat command defaults to the adapter included in this package and the local base-model folder.

## Run in VS Code

1. Install the **Python** extension (Microsoft) in VS Code.
2. **File → Open Folder…** and choose the `JaydariGPT_Local` folder itself (not its parent).
3. Finish the Quick start install steps once (`INSTALL_WINDOWS.bat`, then `download_base_model.py`). The included `.vscode/settings.json` points VS Code at `.venv` automatically; if it asks, pick `.venv\Scripts\python.exe` via **Ctrl+Shift+P → Python: Select Interpreter**.
4. Open **Run and Debug** (Ctrl+Shift+D), pick a configuration and press **F5**:
   - **JaydariGPT: Desktop chat** opens the chat dashboard window
   - **JaydariGPT: Terminal chat** chats inside the VS Code terminal
   - **JaydariGPT: Tokenizer demo** runs the tokenization demo

You can also run from the VS Code terminal (Ctrl+`):

```bat
.venv\Scripts\python.exe gui.py
.venv\Scripts\python.exe main.py chat
```

Model paths resolve against the project folder, so the app also works when started from another directory.

## Desktop chat dashboard

- Sidebar shows the status, model, device (CPU/GPU), load time, message count, last reply time and speed.
- **Creativity** and **Max answer length** sliders change the next answer.
- Answers stream in as they are written; the Send button becomes **Stop** while generating.
- Right-click any message to copy it, or use **Copy last answer**. **New chat** clears the conversation.
- Each message is answered on its own; the model does not remember earlier turns.

## Troubleshooting

**`Base model directory is missing` / local files not found**

Run the one-time download step again while online:

```bat
.venv\Scripts\python.exe download_base_model.py
```

**`python` or `py` is not recognized**

Install Python 3.11 64-bit from https://www.python.org/downloads/ and enable “Add Python to PATH”.

**Out of memory**

Close other applications. CPU inference for a 1.1B model can require several GB of available RAM. A supported NVIDIA GPU can improve speed; 4-bit quantization is used when CUDA and a compatible bitsandbytes installation are available.

**PowerShell blocks virtual-environment activation**

You do not need to activate the environment to launch the app. Use the included `.bat` file or call `.venv\Scripts\python.exe` directly.

## Project layout

```text
JaydariGPT_Local/
├── JaydariGPT.exe                 # Double-click launcher for the dashboard
├── .vscode/                       # VS Code interpreter + Run configurations
├── assets/jaydarigpt.ico          # App icon
├── launcher/                      # Launcher source, icon script and build script
├── main.py
├── gui.py                         # Desktop chat dashboard
├── requirements.txt
├── download_base_model.py
├── INSTALL_WINDOWS.bat
├── START_JAYDARIGPT.bat
├── models/
│   ├── jaydari_gpt_adapter/       # Your supplied trained adapter
│   └── TinyLlama-1.1B-Chat-v1.0/  # Downloaded once during setup
└── src/
    ├── model.py
    ├── inference.py
    ├── prompts.py
    ├── train.py
    └── tokenization.py
```

## Model provenance

The adapter metadata identifies `TinyLlama/TinyLlama-1.1B-Chat-v1.0` as its base model. The adapter weights and tokenizer were copied from the root of your supplied `jaydari_gpt.zip`; checkpoint optimizer and trainer-state files are not required for inference.

Review the base model, dataset, and dependency licenses before redistributing the package or its model weights.
