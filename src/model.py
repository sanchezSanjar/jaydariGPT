"""Local/offline model and tokenizer loading helpers."""
import importlib.util
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ID = "models/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_DIR = "models/jaydari_gpt_adapter"

def resolve_path(path):
    """Resolve a relative path against the working directory, then the project root.

    Lets the app start from any folder (for example a VS Code workspace one level up).
    """
    p = Path(path)
    if p.exists():
        return str(p.resolve())
    if not p.is_absolute() and (PROJECT_ROOT / p).exists():
        return str(PROJECT_ROOT / p)
    return str(path)

_resolve = resolve_path

def can_use_4bit():
    """4-bit loading needs both CUDA and an installed bitsandbytes package."""
    return torch.cuda.is_available() and importlib.util.find_spec("bitsandbytes") is not None

def load_tokenizer(model_id=DEFAULT_MODEL_ID, trust_remote_code=False, local_files_only=True):
    tokenizer = AutoTokenizer.from_pretrained(
        _resolve(model_id), trust_remote_code=trust_remote_code,
        local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model(model_id=DEFAULT_MODEL_ID, use_4bit=False,
                    trust_remote_code=False, for_training=False,
                    local_files_only=True):
    """Load from a local directory by default; no network access is attempted."""
    model_path = _resolve(model_id)
    kwargs = {"trust_remote_code": trust_remote_code,
              "local_files_only": local_files_only}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                raise RuntimeError("4-bit loading unavailable. Install a compatible bitsandbytes build or use --no-4bit.") from exc
    else:
        kwargs["torch_dtype"] = torch.float32
        kwargs["device_map"] = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if for_training:
        model.config.use_cache = False
    return model
