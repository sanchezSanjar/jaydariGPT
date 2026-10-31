"""Desktop chat dashboard for the local JaydariGPT model (Tkinter, stdlib only)."""
from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.inference import load_finetuned_model, stream_answer
from src.model import DEFAULT_ADAPTER_DIR, DEFAULT_MODEL_ID, resolve_path

BASE_MODEL = DEFAULT_MODEL_ID
ADAPTER = DEFAULT_ADAPTER_DIR

BG = "#0f1219"
SIDEBAR = "#151a23"
PANEL = "#1b2130"
PANEL_2 = "#232b3c"
BORDER = "#2c3547"
TEXT = "#edf1f7"
MUTED = "#8f9bb0"
FAINT = "#5f6b80"
ACCENT = "#7c9cff"
ACCENT_HOVER = "#9ab2ff"
USER_BUBBLE = "#2d4a86"
BOT_BUBBLE = "#1e2533"
GREEN = "#6fd3a0"
AMBER = "#e6bd70"
RED = "#e58b8b"

FONT = "Segoe UI"
PLACEHOLDER = "Message JaydariGPT…"
SUGGESTIONS = [
    "Explain machine learning in simple words",
    "Give me three tips for writing clean Python",
    "Write a short email asking for a meeting",
    "What is the difference between RAM and storage?",
]


class JaydariGPTApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("JaydariGPT — Local AI")
        self.root.geometry("1180x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        self.tokenizer = None
        self.model = None
        self.busy = False
        self.stop_event = threading.Event()
        self.events: queue.Queue = queue.Queue()  # worker threads -> Tk main thread
        self.bubbles: list[tk.Label] = []
        self.wrap = 560
        self.message_count = 0
        self.last_answer = ""
        self.current_bubble: tk.Label | None = None
        self.current_meta: tk.Label | None = None
        self.typing_job = None

        self._build_styles()
        self._build_sidebar()
        self._build_main()
        self._show_welcome()
        self.root.after(200, self._load_model)
        self.root.after(50, self._drain_events)

    # ------------------------------------------------------------------ layout
    def _build_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Chat.Vertical.TScrollbar", background=PANEL_2, troughcolor=BG,
                        bordercolor=BG, lightcolor=PANEL_2, darkcolor=PANEL_2,
                        arrowcolor=MUTED, gripcount=0, arrowsize=12)
        style.map("Chat.Vertical.TScrollbar", background=[("active", BORDER)])

    def _build_sidebar(self):
        side = tk.Frame(self.root, bg=SIDEBAR, width=280, padx=22, pady=22)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)

        brand = tk.Frame(side, bg=SIDEBAR)
        brand.pack(fill="x")
        logo = tk.Canvas(brand, width=40, height=40, bg=SIDEBAR, highlightthickness=0)
        logo.create_oval(2, 2, 38, 38, fill=ACCENT, outline="")
        logo.create_text(20, 20, text="J", fill="#0e1220", font=(FONT, 15, "bold"))
        logo.pack(side="left")
        names = tk.Frame(brand, bg=SIDEBAR)
        names.pack(side="left", padx=(12, 0))
        tk.Label(names, text="JaydariGPT", bg=SIDEBAR, fg=TEXT,
                 font=(FONT, 15, "bold")).pack(anchor="w")
        tk.Label(names, text="Private AI · runs on this PC", bg=SIDEBAR, fg=MUTED,
                 font=(FONT, 9)).pack(anchor="w")

        status_card = tk.Frame(side, bg=PANEL, padx=14, pady=11)
        status_card.pack(fill="x", pady=(22, 0))
        self.status_dot = tk.Canvas(status_card, width=10, height=10, bg=PANEL,
                                    highlightthickness=0)
        self.status_dot_id = self.status_dot.create_oval(1, 1, 9, 9, fill=AMBER, outline="")
        self.status_dot.pack(side="left")
        self.status = tk.Label(status_card, text="Starting…", bg=PANEL, fg=TEXT,
                               font=(FONT, 10))
        self.status.pack(side="left", padx=(10, 0))

        self.new_chat_btn = self._button(side, "＋  New chat", self.clear_chat,
                                         bg=PANEL_2, fg=TEXT, hover=BORDER)
        self.new_chat_btn.pack(fill="x", pady=(14, 0))

        self._section(side, "MODEL")
        self.stat_base = self._stat_row(side, "Base", "TinyLlama 1.1B Chat")
        self.stat_adapter = self._stat_row(side, "Adapter", self._adapter_summary())
        self.stat_device = self._stat_row(side, "Device", "—")
        self.stat_load = self._stat_row(side, "Load time", "—")

        self._section(side, "SESSION")
        self.stat_messages = self._stat_row(side, "Messages", "0")
        self.stat_reply = self._stat_row(side, "Last reply", "—")
        self.stat_speed = self._stat_row(side, "Speed", "—")

        self._section(side, "SETTINGS")
        self.temperature = tk.DoubleVar(value=0.7)
        self.max_tokens = tk.IntVar(value=256)
        self._slider(side, "Creativity (temperature)", self.temperature, 0.0, 1.5, 0.05,
                     lambda v: f"{float(v):.2f}")
        self._slider(side, "Max answer length (tokens)", self.max_tokens, 32, 512, 32,
                     lambda v: f"{int(float(v))}")

        tk.Label(side, text="Offline after setup\nYour messages never leave this PC",
                 bg=SIDEBAR, fg=FAINT, font=(FONT, 8), justify="left").pack(
            side="bottom", anchor="w")

    def _build_main(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=BG, padx=28, pady=16)
        header.pack(fill="x")
        titles = tk.Frame(header, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text="Chat", bg=BG, fg=TEXT, font=(FONT, 14, "bold")).pack(anchor="w")
        tk.Label(titles, text="Each message is answered on its own (no memory of earlier turns)",
                 bg=BG, fg=FAINT, font=(FONT, 9)).pack(anchor="w")
        self.copy_btn = self._button(header, "Copy last answer", self.copy_last_answer,
                                     bg=PANEL, fg=MUTED, hover=PANEL_2, font_size=9)
        self.copy_btn.pack(side="right")
        tk.Frame(main, bg=BORDER, height=1).pack(fill="x")

        feed_wrap = tk.Frame(main, bg=BG)
        feed_wrap.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(feed_wrap, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(feed_wrap, orient="vertical", command=self.canvas.yview,
                                  style="Chat.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.feed = tk.Frame(self.canvas, bg=BG)
        self.feed_id = self.canvas.create_window((0, 0), window=self.feed, anchor="nw")
        self.feed.bind("<Configure>", lambda _e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Enter>", lambda _e: self.root.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind("<Leave>", lambda _e: self.root.unbind_all("<MouseWheel>"))

        composer_wrap = tk.Frame(main, bg=BG, padx=28, pady=14)
        composer_wrap.pack(fill="x")
        composer = tk.Frame(composer_wrap, bg=PANEL_2, padx=14, pady=10,
                            highlightthickness=1, highlightbackground=BORDER,
                            highlightcolor=ACCENT)
        composer.pack(fill="x")
        self.input = tk.Text(composer, height=3, wrap="word", bg=PANEL_2, fg=TEXT,
                             insertbackground=TEXT, relief="flat", bd=0,
                             font=(FONT, 11), selectbackground="#3a4e73")
        self.input.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.input.bind("<Return>", self._enter_to_send)
        self.input.bind("<FocusIn>", self._clear_placeholder)
        self.input.bind("<FocusOut>", self._restore_placeholder)
        self._restore_placeholder()
        self.send_btn = self._button(composer, "Send  ➤", self._send_or_stop,
                                     bg=ACCENT, fg="#101522", hover=ACCENT_HOVER, bold=True)
        self.send_btn.pack(side="right", anchor="s")
        self.send_btn.configure(state="disabled")
        tk.Label(composer_wrap, text="Enter to send  ·  Shift+Enter for a new line",
                 bg=BG, fg=FAINT, font=(FONT, 8)).pack(anchor="w", pady=(6, 0))

    # ----------------------------------------------------------------- widgets
    def _button(self, parent, text, command, bg, fg, hover, bold=False, font_size=10):
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                        activebackground=hover, activeforeground=fg, relief="flat", bd=0,
                        padx=16, pady=9, cursor="hand2",
                        font=(FONT, font_size, "bold" if bold else "normal"),
                        disabledforeground=FAINT)
        btn.bind("<Enter>", lambda _e: btn.configure(bg=hover) if btn["state"] != "disabled" else None)
        btn.bind("<Leave>", lambda _e: btn.configure(bg=bg))
        return btn

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=SIDEBAR, fg=FAINT,
                 font=(FONT, 8, "bold")).pack(anchor="w", pady=(24, 6))

    def _stat_row(self, parent, label, value):
        row = tk.Frame(parent, bg=SIDEBAR)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, bg=SIDEBAR, fg=MUTED, font=(FONT, 9)).pack(side="left")
        value_label = tk.Label(row, text=value, bg=SIDEBAR, fg=TEXT, font=(FONT, 9))
        value_label.pack(side="right")
        return value_label

    def _slider(self, parent, label, variable, low, high, step, fmt):
        top = tk.Frame(parent, bg=SIDEBAR)
        top.pack(fill="x", pady=(4, 0))
        tk.Label(top, text=label, bg=SIDEBAR, fg=MUTED, font=(FONT, 9)).pack(side="left")
        value = tk.Label(top, text=fmt(variable.get()), bg=SIDEBAR, fg=TEXT, font=(FONT, 9))
        value.pack(side="right")
        tk.Scale(parent, from_=low, to=high, resolution=step, orient="horizontal",
                 variable=variable, showvalue=False, bg=ACCENT, troughcolor=PANEL_2,
                 activebackground=ACCENT_HOVER, highlightthickness=0, bd=0,
                 sliderrelief="flat", sliderlength=16, width=10, cursor="hand2",
                 command=lambda v: value.configure(text=fmt(v))).pack(fill="x", pady=(2, 6))

    def _adapter_summary(self):
        try:
            config = json.loads((Path(resolve_path(ADAPTER)) / "adapter_config.json").read_text())
            return f"LoRA · rank {config.get('r', '?')}"
        except (OSError, ValueError):
            return "LoRA"

    # -------------------------------------------------------------------- feed
    def _show_welcome(self):
        self.welcome = tk.Frame(self.feed, bg=BG)
        self.welcome.pack(fill="x", pady=(90, 20))
        tk.Label(self.welcome, text="How can I help today?", bg=BG, fg=TEXT,
                 font=(FONT, 20, "bold")).pack()
        tk.Label(self.welcome, text="Pick an example or type your own question below.",
                 bg=BG, fg=MUTED, font=(FONT, 10)).pack(pady=(6, 22))
        grid = tk.Frame(self.welcome, bg=BG)
        grid.pack()
        grid.columnconfigure((0, 1), minsize=290, uniform="card")
        self.suggestion_buttons = []
        for index, text in enumerate(SUGGESTIONS):
            btn = tk.Button(grid, text=text, command=lambda t=text: self.send_message(t),
                            bg=PANEL, fg=TEXT, activebackground=PANEL_2, activeforeground=TEXT,
                            disabledforeground=FAINT, relief="flat", bd=0, padx=16, pady=14,
                            wraplength=250, justify="left", anchor="w",
                            cursor="hand2", font=(FONT, 10),
                            state="normal" if self.model is not None else "disabled")
            btn.grid(row=index // 2, column=index % 2, padx=6, pady=6, sticky="nsew")
            btn.bind("<Enter>", lambda _e, b=btn: b.configure(bg=PANEL_2) if b["state"] != "disabled" else None)
            btn.bind("<Leave>", lambda _e, b=btn: b.configure(bg=PANEL))
            self.suggestion_buttons.append(btn)

    def _add_message(self, text: str, is_user: bool):
        if self.welcome is not None:
            self.welcome.destroy()
            self.welcome = None
        row = tk.Frame(self.feed, bg=BG)
        row.pack(fill="x", padx=28, pady=8)
        meta = None
        if is_user:
            column = tk.Frame(row, bg=BG)
            column.pack(side="right")
            bubble = tk.Label(column, text=text, bg=USER_BUBBLE, fg=TEXT, font=(FONT, 11),
                              justify="left", anchor="w", wraplength=self.wrap, padx=16, pady=10)
            bubble.pack(anchor="e")
        else:
            avatar = tk.Canvas(row, width=32, height=32, bg=BG, highlightthickness=0)
            avatar.create_oval(1, 1, 31, 31, fill=ACCENT, outline="")
            avatar.create_text(16, 16, text="J", fill="#0e1220", font=(FONT, 11, "bold"))
            avatar.pack(side="left", anchor="n", padx=(0, 12))
            column = tk.Frame(row, bg=BG)
            column.pack(side="left", fill="x")
            tk.Label(column, text="JaydariGPT", bg=BG, fg=GREEN,
                     font=(FONT, 9, "bold")).pack(anchor="w", pady=(0, 4))
            bubble = tk.Label(column, text=text, bg=BOT_BUBBLE, fg=TEXT, font=(FONT, 11),
                              justify="left", anchor="w", wraplength=self.wrap, padx=16, pady=10)
            bubble.pack(anchor="w")
            meta = tk.Label(column, text="", bg=BG, fg=FAINT, font=(FONT, 8))
            meta.pack(anchor="w", pady=(4, 0))
        bubble.bind("<Button-3>", lambda _e, b=bubble: self._copy(b.cget("text")))
        self.bubbles.append(bubble)
        self._scroll_to_end()
        return bubble, meta

    def _add_notice(self, text: str, color: str = MUTED):
        if self.welcome is not None:
            self.welcome.destroy()
            self.welcome = None
        tk.Label(self.feed, text=text, bg=BG, fg=color, font=(FONT, 9),
                 wraplength=self.wrap).pack(pady=10)
        self._scroll_to_end()

    def _on_resize(self, event):
        self.canvas.itemconfigure(self.feed_id, width=event.width)
        self.wrap = max(280, min(720, int(event.width * 0.66)))
        for bubble in self.bubbles:
            bubble.configure(wraplength=self.wrap)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _scroll_to_end(self):
        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def _animate_typing(self, step=0):
        if self.current_bubble is None or self.current_bubble.cget("text") not in ("", "●", "● ●", "● ● ●"):
            return
        self.current_bubble.configure(text=["●", "● ●", "● ● ●"][step % 3], fg=MUTED)
        self.typing_job = self.root.after(350, self._animate_typing, step + 1)

    # ----------------------------------------------------------------- status
    def _set_status(self, message: str, color: str):
        self.status.configure(text=message)
        self.status_dot.itemconfigure(self.status_dot_id, fill=color)

    # ------------------------------------------------------------ model work
    def _load_model(self):
        missing = [p for p in (BASE_MODEL, ADAPTER) if not Path(resolve_path(p)).exists()]
        if missing:
            self._set_status("Model files missing", RED)
            self._add_notice("Model files are missing: " + ", ".join(missing) +
                             ". Run INSTALL_WINDOWS.bat and download_base_model.py first.", RED)
            messagebox.showwarning(
                "Model not found",
                "JaydariGPT model files were not found.\n\n"
                "Run the installation and base-model download steps in README.md, then restart the app."
            )
            return
        self._set_status("Loading model…", AMBER)
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        started = time.perf_counter()
        try:
            tokenizer, model = load_finetuned_model(BASE_MODEL, ADAPTER)
            device = next(model.parameters()).device
            self.events.put(("loaded", tokenizer, model, time.perf_counter() - started,
                             "GPU (CUDA)" if device.type == "cuda" else "CPU"))
        except Exception as exc:
            self.events.put(("load_error", str(exc)))

    def _generate_worker(self, prompt: str, max_new_tokens: int, temperature: float):
        started = time.perf_counter()
        text = ""
        try:
            for text in stream_answer(self.tokenizer, self.model, prompt,
                                      max_new_tokens=max_new_tokens, temperature=temperature,
                                      top_p=0.9, stop_event=self.stop_event):
                self.events.put(("token", text))
            tokens = len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
            self.events.put(("done", text.strip(), time.perf_counter() - started, tokens,
                             self.stop_event.is_set()))
        except Exception as exc:
            self.events.put(("gen_error", str(exc)))

    def _drain_events(self):
        """Apply worker results on the Tk thread; Tk widgets are not thread-safe."""
        try:
            while True:
                kind, *data = self.events.get_nowait()
                getattr(self, "_on_" + kind)(*data)
        except queue.Empty:
            pass
        self.root.after(50, self._drain_events)

    def _on_loaded(self, tokenizer, model, seconds, device):
        self.tokenizer, self.model = tokenizer, model
        self.stat_device.configure(text=device)
        self.stat_load.configure(text=f"{seconds:.1f} s")
        self._set_status("Ready · offline", GREEN)
        self.send_btn.configure(state="normal")
        for btn in getattr(self, "suggestion_buttons", []):
            if btn.winfo_exists():
                btn.configure(state="normal")
        self.input.focus_set()

    def _on_load_error(self, error):
        self._set_status("Could not load model", RED)
        self._add_notice("Model loading failed. Check the setup steps in README.md.", RED)
        messagebox.showerror("Could not load JaydariGPT", error[:3000])

    def _on_token(self, text):
        if self.current_bubble is not None:
            self.current_bubble.configure(text=text.strip() or "● ● ●", fg=TEXT)
            self._scroll_to_end()

    def _on_done(self, text, seconds, tokens, stopped):
        if not text:
            text = "I couldn't generate a response. Please try again."
        self.current_bubble.configure(text=text, fg=TEXT)
        speed = tokens / seconds if seconds > 0 else 0
        note = f"{seconds:.1f} s  ·  {speed:.1f} tokens/s" + ("  ·  stopped" if stopped else "")
        self.current_meta.configure(text=note)
        self.last_answer = text
        self.stat_reply.configure(text=f"{seconds:.1f} s")
        self.stat_speed.configure(text=f"{speed:.1f} tok/s")
        self._finish()

    def _on_gen_error(self, error):
        self.current_bubble.configure(text="Generation error: " + error[:1200], fg=RED)
        self._finish(error=True)

    def _finish(self, error=False):
        self.busy = False
        self.current_bubble = self.current_meta = None
        self.send_btn.configure(state="normal", text="Send  ➤")
        self.new_chat_btn.configure(state="normal")
        self._set_status("Ready · check error" if error else "Ready · offline",
                         AMBER if error else GREEN)
        self._scroll_to_end()
        self.input.focus_set()

    # ---------------------------------------------------------------- actions
    def _send_or_stop(self):
        if self.busy:
            self.stop_event.set()
            self.send_btn.configure(text="Stopping…", state="disabled")
        else:
            self.send_message()

    def send_message(self, prompt: str | None = None):
        if self.busy or self.model is None:
            return
        if prompt is None:
            prompt = self.input.get("1.0", "end-1c").strip()
            if not prompt or prompt == PLACEHOLDER:
                return
            self.input.delete("1.0", "end")
        self._add_message(prompt, is_user=True)
        self.current_bubble, self.current_meta = self._add_message("", is_user=False)
        self._animate_typing()
        self.message_count += 1
        self.stat_messages.configure(text=str(self.message_count))
        self.busy = True
        self.stop_event.clear()
        self.send_btn.configure(text="■  Stop")
        self.new_chat_btn.configure(state="disabled")
        self._set_status("Generating…", AMBER)
        threading.Thread(target=self._generate_worker,
                         args=(prompt, self.max_tokens.get(), self.temperature.get()),
                         daemon=True).start()

    def _enter_to_send(self, event):
        if not (event.state & 0x0001):  # Shift+Enter inserts a newline
            if not self.busy:
                self.send_message()
            return "break"
        return None

    def _clear_placeholder(self, _event=None):
        if self.input.get("1.0", "end-1c") == PLACEHOLDER:
            self.input.delete("1.0", "end")
            self.input.configure(fg=TEXT)

    def _restore_placeholder(self, _event=None):
        if not self.input.get("1.0", "end-1c").strip():
            self.input.delete("1.0", "end")
            self.input.insert("1.0", PLACEHOLDER)
            self.input.configure(fg=FAINT)

    def _copy(self, text):
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self._set_status("Copied to clipboard", GREEN)
            self.root.after(1500, lambda: self._set_status(
                "Generating…" if self.busy else "Ready · offline", AMBER if self.busy else GREEN))

    def copy_last_answer(self):
        self._copy(self.last_answer)

    def clear_chat(self):
        if self.busy:
            return
        for child in self.feed.winfo_children():
            child.destroy()
        self.bubbles.clear()
        self.welcome = None
        self.last_answer = ""
        self.message_count = 0
        self.stat_messages.configure(text="0")
        self._show_welcome()
        self.canvas.yview_moveto(0)


def main():
    root = tk.Tk()
    JaydariGPTApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
