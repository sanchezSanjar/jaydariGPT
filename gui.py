"""Desktop chat interface for the local JaydariGPT model (Tkinter, stdlib only)."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from pathlib import Path

from src.inference import load_finetuned_model, generate_answer

BASE_MODEL = "models/TinyLlama-1.1B-Chat-v1.0"
ADAPTER = "models/jaydari_gpt_adapter"

BG = "#10131a"
PANEL = "#191e29"
PANEL_2 = "#222938"
TEXT = "#edf1f7"
MUTED = "#9aa6b8"
ACCENT = "#7c9cff"
USER_BUBBLE = "#263858"
BOT_BUBBLE = "#202733"


class JaydariGPTApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JaydariGPT — Local AI")
        self.root.geometry("1000x760")
        self.root.minsize(720, 560)
        self.root.configure(bg=BG)
        self.tokenizer = None
        self.model = None
        self.busy = False
        self._build_ui()
        self.root.after(250, self._load_model)

    def _build_ui(self):
        header = tk.Frame(self.root, bg=PANEL, padx=22, pady=15)
        header.pack(fill="x")

        brand = tk.Frame(header, bg=PANEL)
        brand.pack(side="left", fill="x", expand=True)
        tk.Label(brand, text="JaydariGPT", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 19, "bold")).pack(anchor="w")
        tk.Label(brand, text="YOUR PRIVATE AI  ·  RUNNING LOCALLY",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(2, 0))

        self.status = tk.Label(header, text="●  Starting…", bg=PANEL, fg="#e6bd70",
                               font=("Segoe UI", 9))
        self.status.pack(side="right", padx=(10, 0))

        toolbar = tk.Frame(self.root, bg=BG, padx=20, pady=10)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Chat", bg=BG, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(side="left")
        self.clear_btn = tk.Button(toolbar, text="Clear conversation", command=self.clear_chat,
                                   bg=PANEL_2, fg=MUTED, activebackground="#303b50",
                                   activeforeground=TEXT, relief="flat", bd=0,
                                   padx=12, pady=7, cursor="hand2", font=("Segoe UI", 9))
        self.clear_btn.pack(side="right")

        chat_frame = tk.Frame(self.root, bg=BG, padx=20)
        chat_frame.pack(fill="both", expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_frame, wrap="word", state="disabled", bg=BG, fg=TEXT,
            insertbackground=TEXT, selectbackground="#3a4e73", relief="flat",
            padx=12, pady=12, font=("Segoe UI", 11), spacing1=4, spacing3=9,
            borderwidth=0
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("user_name", foreground="#a9c1ff", font=("Segoe UI", 10, "bold"), spacing1=12)
        self.chat.tag_configure("bot_name", foreground="#91d8b5", font=("Segoe UI", 10, "bold"), spacing1=12)
        self.chat.tag_configure("body", foreground=TEXT, lmargin1=12, lmargin2=12, rmargin=16)
        self.chat.tag_configure("system", foreground=MUTED, font=("Segoe UI", 9), justify="center", spacing1=10, spacing3=10)
        self._append_system("Loading your local model. The first startup can take a few minutes.")

        composer = tk.Frame(self.root, bg=PANEL, padx=16, pady=14)
        composer.pack(fill="x", padx=20, pady=(12, 18))
        self.input = tk.Text(composer, height=3, wrap="word", bg=PANEL_2, fg=TEXT,
                             insertbackground=TEXT, relief="flat", bd=0,
                             padx=12, pady=10, font=("Segoe UI", 11),
                             highlightthickness=1, highlightbackground="#30394a",
                             highlightcolor=ACCENT)
        self.input.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.input.bind("<Return>", self._enter_to_send)
        self.input.bind("<Shift-Return>", lambda _e: None)
        self.send_btn = tk.Button(composer, text="Send  ➤", command=self.send_message,
                                  bg=ACCENT, fg="#101522", activebackground="#9ab2ff",
                                  activeforeground="#101522", relief="flat", bd=0,
                                  padx=18, pady=13, cursor="hand2",
                                  font=("Segoe UI", 10, "bold"), state="disabled")
        self.send_btn.pack(side="right", anchor="s")

        footer = tk.Label(self.root, text="Offline after setup  ·  Your messages stay on this PC",
                          bg=BG, fg="#748095", font=("Segoe UI", 8))
        footer.pack(pady=(0, 8))

    def _append(self, speaker: str, message: str, is_user: bool):
        self.chat.configure(state="normal")
        self.chat.insert("end", speaker + "\n", "user_name" if is_user else "bot_name")
        self.chat.insert("end", message.strip() + "\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _append_system(self, message: str):
        self.chat.configure(state="normal")
        self.chat.insert("end", message + "\n", "system")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _set_status(self, message: str, color: str = MUTED):
        self.status.configure(text="●  " + message, fg=color)

    def _load_model(self):
        missing = [p for p in (BASE_MODEL, ADAPTER) if not Path(p).exists()]
        if missing:
            self._set_status("Model files missing", "#e6bd70")
            self._append_system("Model files are missing. Complete INSTALL_WINDOWS.bat and the one-time base-model download first.")
            messagebox.showwarning(
                "Model not found",
                "JaydariGPT model files were not found.\n\n"
                "Run the installation and base-model download steps in README.md, then restart the app."
            )
            return
        self._set_status("Loading model…", "#e6bd70")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            tokenizer, model = load_finetuned_model(BASE_MODEL, ADAPTER)
            self.tokenizer, self.model = tokenizer, model
            self.root.after(0, self._model_ready)
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._model_error(error))

    def _model_ready(self):
        self._set_status("Ready · Offline", "#83d6a8")
        self.send_btn.configure(state="normal")
        self._append_system("JaydariGPT is ready. Ask a question to start chatting.")
        self.input.focus_set()

    def _model_error(self, error: str):
        self._set_status("Could not load model", "#e58b8b")
        self._append_system("Model loading failed. Check the setup instructions and error details.")
        messagebox.showerror("Could not load JaydariGPT", error[:3000])

    def _enter_to_send(self, event):
        if not (event.state & 0x0001):  # Shift+Enter inserts a newline
            self.send_message()
            return "break"
        return None

    def send_message(self):
        if self.busy or self.model is None:
            return
        prompt = self.input.get("1.0", "end-1c").strip()
        if not prompt:
            return
        self.input.delete("1.0", "end")
        self._append("You", prompt, True)
        self.busy = True
        self.send_btn.configure(state="disabled", text="Thinking…")
        self._set_status("Generating response…", "#e6bd70")
        threading.Thread(target=self._generate_worker, args=(prompt,), daemon=True).start()

    def _generate_worker(self, prompt: str):
        try:
            answer = generate_answer(self.tokenizer, self.model, prompt,
                                     max_new_tokens=256, temperature=0.7, top_p=0.9)
            if not answer:
                answer = "I couldn't generate a response. Please try again."
            self.root.after(0, lambda: self._show_answer(answer))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._generation_error(error))

    def _show_answer(self, answer: str):
        self._append("JaydariGPT", answer, False)
        self._finish_generation()

    def _generation_error(self, error: str):
        self._append_system("Generation error: " + error[:1200])
        self._finish_generation(error=True)

    def _finish_generation(self, error=False):
        self.busy = False
        self.send_btn.configure(state="normal", text="Send  ➤")
        self._set_status("Ready · Offline" if not error else "Ready · Check error", "#83d6a8" if not error else "#e6bd70")
        self.input.focus_set()

    def clear_chat(self):
        if self.busy:
            return
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._append_system("Conversation cleared. JaydariGPT is ready.")


def main():
    root = tk.Tk()
    JaydariGPTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
