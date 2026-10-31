"""Small educational tokenizer and TinyLlama tokenizer demonstration."""
import torch
from torch import nn
from transformers import AutoTokenizer
from .model import DEFAULT_MODEL_ID, resolve_path


def run_demo():
    sample = (
        "Hello world this is a simple example to show how tokenization "
        "works in NLP and a sentence"
    )
    words = sample.split()
    vocab = {}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)
    encoded = [vocab[word] for word in words]
    reverse_vocab = {index: word for word, index in vocab.items()}
    decoded = " ".join(reverse_vocab[index] for index in encoded)

    print("Sample text:", sample)
    print("Word-level tokens:", words)
    print("Vocabulary:", vocab)
    print("Encoded IDs:", encoded)
    print("Decoded text:", decoded)

    embedding = nn.Embedding(len(vocab), 8)
    input_ids = torch.tensor([encoded], dtype=torch.long)
    print("Embedding shape:", embedding(input_ids).shape)

    print("\n--- TinyLlama tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(resolve_path(DEFAULT_MODEL_ID), local_files_only=True)
    example = "Hello, how are you doing?"
    token_ids = tokenizer(example, return_tensors="pt")
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Special tokens:", tokenizer.special_tokens_map)
    print("Tokens:", tokenizer.tokenize(example))
    print("Input IDs:", token_ids["input_ids"][0].tolist())
    print("Decoded:", tokenizer.decode(token_ids["input_ids"][0]))
