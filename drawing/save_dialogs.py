"""Native file pickers for exact Hansa save files."""

from __future__ import annotations

from pathlib import Path

from game.persistence import SAVE_EXTENSION, default_save_directory, suggested_save_name


def _dialog_root():
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    return root


def _initial_directory() -> Path:
    directory = default_save_directory()
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path.home()
    return directory


def choose_save_file(game) -> Path | None:
    from tkinter import filedialog

    directory = _initial_directory()
    root = _dialog_root()
    try:
        filename = filedialog.asksaveasfilename(
            parent=root,
            title="Save Hansa Game",
            initialdir=directory,
            initialfile=suggested_save_name(game),
            defaultextension=SAVE_EXTENSION,
            filetypes=(("Hansa saved games", f"*{SAVE_EXTENSION}"),),
        )
    finally:
        root.destroy()
    return Path(filename) if filename else None


def choose_load_file() -> Path | None:
    from tkinter import filedialog

    directory = _initial_directory()
    root = _dialog_root()
    try:
        filename = filedialog.askopenfilename(
            parent=root,
            title="Load Hansa Game",
            initialdir=directory,
            filetypes=(("Hansa saved games", f"*{SAVE_EXTENSION}"),),
        )
    finally:
        root.destroy()
    return Path(filename) if filename else None
