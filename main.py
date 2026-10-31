"""Command-line entry point for JaydariGPT."""
import argparse
from src.train import train
from src.inference import chat
from src.tokenization import run_demo


def build_parser():
    parser = argparse.ArgumentParser(
        prog="jaydariGPT",
        description="Train and run the JaydariGPT TinyLlama LoRA adapter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune the base model.")
    train_parser.add_argument("--model", default="models/TinyLlama-1.1B-Chat-v1.0")
    train_parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    train_parser.add_argument("--split", default="train")
    train_parser.add_argument("--max-samples", type=int, default=7000)
    train_parser.add_argument("--epochs", type=float, default=2)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--max-length", type=int, default=512)
    # Kept separate from the shipped adapter so a training run never overwrites it.
    train_parser.add_argument("--output-dir", default="models/jaydari_gpt_adapter_new")
    train_parser.add_argument("--no-4bit", action="store_true",
                              help="Do not use 4-bit quantization (uses more VRAM/RAM).")
    train_parser.add_argument("--trust-remote-code", action="store_true")

    chat_parser = subparsers.add_parser("chat", help="Chat with a saved adapter.")
    chat_parser.add_argument("--adapter", default="models/jaydari_gpt_adapter")
    chat_parser.add_argument("--model", default="models/TinyLlama-1.1B-Chat-v1.0")
    chat_parser.add_argument("--max-new-tokens", type=int, default=256)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument("--top-p", type=float, default=0.9)
    chat_parser.add_argument("--allow-non-english", action="store_true",
                             help="Disable the original English-only prompt check.")
    chat_parser.add_argument("--trust-remote-code", action="store_true")

    subparsers.add_parser("tokenize", help="Run the tokenizer and embedding demo.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "chat":
        chat(args)
    elif args.command == "tokenize":
        run_demo()


if __name__ == "__main__":
    main()
