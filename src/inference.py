"""Offline interactive inference using the included PEFT/LoRA adapter."""
import torch
from peft import PeftModel
from .model import DEFAULT_MODEL_ID, can_use_4bit, load_base_model, load_tokenizer
from .prompts import STOP_STRING, format_chat_prompt

def load_finetuned_model(model_id, adapter_dir, trust_remote_code=False):
    tokenizer = load_tokenizer(adapter_dir, trust_remote_code, local_files_only=True)
    base_model = load_base_model(
        model_id, use_4bit=can_use_4bit(),
        trust_remote_code=trust_remote_code, local_files_only=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return tokenizer, model

def generate_answer(tokenizer, model, prompt, max_new_tokens=256,
                    temperature=0.7, top_p=0.9):
    formatted_prompt = format_chat_prompt(prompt)
    device = next(model.parameters()).device
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5),
        "top_p": top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        # Stop when the model starts inventing the next "### Instruction:" block.
        "stop_strings": [STOP_STRING],
        "tokenizer": tokenizer,
    }
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_args)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer.split(STOP_STRING, 1)[0].strip()

def chat(args):
    tokenizer, model = load_finetuned_model(args.model, args.adapter, args.trust_remote_code)
    print("\nJaydariGPT is ready (local/offline mode). Type 'exit' or 'quit' to stop.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not prompt:
            continue
        answer = generate_answer(tokenizer, model, prompt, args.max_new_tokens,
                                 args.temperature, args.top_p)
        print(f"\nJaydariGPT: {answer}")
