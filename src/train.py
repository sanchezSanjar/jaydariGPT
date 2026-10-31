"""Dataset preparation and LoRA fine-tuning."""
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from .model import DEFAULT_MODEL_ID, can_use_4bit, load_base_model, load_tokenizer
from .prompts import format_instruction


def prepare_dataset(dataset_name, split, tokenizer, max_samples, max_length):
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=42)
    if max_samples and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize_batch(batch):
        # The trailing EOS teaches the model where an answer ends.
        texts = [
            format_instruction(inst, inp, out) + tokenizer.eos_token
            for inst, inp, out in zip(
                batch["instruction"], batch.get("input", [""] * len(batch["instruction"])),
                batch["output"]
            )
        ]
        return tokenizer(
            texts, truncation=True, max_length=max_length, padding=False
        )

    required = {"instruction", "output"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    tokenized = dataset.map(
        tokenize_batch, batched=True, remove_columns=dataset.column_names,
        desc="Tokenizing training examples"
    )
    return tokenized


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.model, args.trust_remote_code)
    # TinyLlama pads with EOS, and the collator masks pad tokens out of the loss,
    # which would hide every EOS from training. Pad with UNK instead.
    if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    use_4bit = not args.no_4bit and can_use_4bit()
    if not args.no_4bit and torch.cuda.is_available() and not use_4bit:
        print("bitsandbytes is not installed; training without 4-bit quantization.")
    model = load_base_model(
        args.model, use_4bit=use_4bit,
        trust_remote_code=args.trust_remote_code, for_training=True
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = prepare_dataset(
        args.dataset, args.split, tokenizer, args.max_samples, args.max_length
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=use_4bit,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTraining complete. LoRA adapter saved to: {args.output_dir}")
