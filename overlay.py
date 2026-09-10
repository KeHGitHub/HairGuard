"""Temporary HairGuard warning window, launched separately from detection."""

from __future__ import annotations

import sys
import tkinter as tk


def show_stop_overlay() -> None:
    """Show the warning until HairGuard closes it or the user presses a key."""
    root = tk.Tk()

    def dismiss(_event: object = None) -> None:
        root.destroy()

    root.title("HairGuard")
    root.configure(background="#d7193f")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.protocol("WM_DELETE_WINDOW", dismiss)
    root.bind("<Key>", dismiss)

    tk.Label(
        root,
        text="STOP",
        font=("Helvetica Neue", 104, "bold"),
        foreground="white",
        background="#d7193f",
    ).pack(expand=True)
    tk.Label(
        root,
        text="Move your hand away from your head  •  Any key dismisses",
        font=("Helvetica Neue", 18),
        foreground="white",
        background="#d7193f",
    ).pack(pady=(0, 60))

    root.lift()
    root.focus_force()
    root.mainloop()


if __name__ == "__main__":
    try:
        show_stop_overlay()
    except tk.TclError as error:
        print(f"HairGuard overlay error: {error}", file=sys.stderr)
        raise SystemExit(1)
