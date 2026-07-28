"""
Common GUI widgets and helper functions used across all GUIs.

This module provides reusable widget components to reduce code duplication
and ensure consistency across the Individual, Parameter Sweep, and Genotype Sweep GUIs.
"""

from __future__ import annotations

import sys
import tkinter as tk


_TIP_BG = "#ffffe0"
_TIP_BORDER = "#a0a000"
_TIP_WRAP_PX = 480
# Inset on macOS so system-rounded tooltip window corners do not clip text.
_TIP_WINDOW_PAD = 8 if sys.platform == "darwin" else 4
_TIP_INNER_PAD_X = 10
_TIP_INNER_PAD_Y = 8


class CreateToolTip:
    """
    Tooltip widget that displays help text when hovering over a widget.

    Uses a short show delay and a brief hide grace period so moving between
    adjacent controls (label + entry) does not flash the tip. Positions the tip
    beside the widget instead of under the pointer to avoid Leave/Enter loops
    when the tip window appears.

    Usage:
        CreateToolTip(my_widget, "This is helpful text")
    """

    _SHOW_DELAY_MS = 450
    _HIDE_DELAY_MS = 250

    def __init__(self, widget, text):
        self.widget = widget
        self.text = str(text or "").strip()
        self.tip_window = None
        self._show_after_id = None
        self._hide_after_id = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _cancel_show(self) -> None:
        if self._show_after_id is not None:
            try:
                self.widget.after_cancel(self._show_after_id)
            except tk.TclError:
                pass
            self._show_after_id = None

    def _cancel_hide(self) -> None:
        if self._hide_after_id is not None:
            try:
                self.widget.after_cancel(self._hide_after_id)
            except tk.TclError:
                pass
            self._hide_after_id = None

    def _on_enter(self, _event=None) -> None:
        self._cancel_hide()
        if self.tip_window or not self.text:
            return
        if self._show_after_id is not None:
            return
        self._show_after_id = self.widget.after(self._SHOW_DELAY_MS, self._show_tip)

    def _on_leave(self, _event=None) -> None:
        self._cancel_show()
        if not self.tip_window:
            return
        self._cancel_hide()
        self._hide_after_id = self.widget.after(self._HIDE_DELAY_MS, self._hide_tip)

    def _tip_geometry(self) -> tuple[int, int]:
        try:
            if not self.widget.winfo_exists():
                return 0, 0
        except tk.TclError:
            return 0, 0

        root_x = int(self.widget.winfo_rootx())
        root_y = int(self.widget.winfo_rooty())
        width = max(int(self.widget.winfo_width()), 1)
        height = max(int(self.widget.winfo_height()), 1)

        # Place to the right of the widget so the tip does not sit under the cursor.
        x = root_x + width + 8
        y = root_y + max(0, (height // 2) - 10)

        try:
            bbox = self.widget.bbox("insert")
            if bbox:
                x = root_x + int(bbox[0]) + int(bbox[2]) + 12
                y = root_y + int(bbox[1]) + int(bbox[3]) + 4
        except tk.TclError:
            pass

        return x, y

    def _show_tip(self) -> None:
        self._show_after_id = None
        if self.tip_window or not self.text:
            return
        try:
            if not self.widget.winfo_exists():
                return
        except tk.TclError:
            return

        x, y = self._tip_geometry()
        if x == 0 and y == 0:
            return

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.configure(bg=_TIP_BG)
        try:
            tw.wm_attributes("-topmost", True)
        except tk.TclError:
            pass

        shell = tk.Frame(
            tw,
            bg=_TIP_BG,
            bd=0,
            highlightthickness=1,
            highlightbackground=_TIP_BORDER,
            highlightcolor=_TIP_BORDER,
        )
        shell.pack(padx=_TIP_WINDOW_PAD, pady=_TIP_WINDOW_PAD)

        label = tk.Label(
            shell,
            text=self.text,
            justify="left",
            anchor="nw",
            background=_TIP_BG,
            foreground="#000000",
            relief="flat",
            borderwidth=0,
            font=("tahoma", 9, "normal"),
            wraplength=_TIP_WRAP_PX,
        )
        label.pack(padx=_TIP_INNER_PAD_X, pady=_TIP_INNER_PAD_Y, anchor="nw")

        tw.update_idletasks()
        tw.wm_geometry(f"+{x}+{y}")

    def _hide_tip(self) -> None:
        self._hide_after_id = None
        if not self.tip_window:
            return
        try:
            self.tip_window.destroy()
        except tk.TclError:
            pass
        self.tip_window = None

    def _on_destroy(self, _event=None) -> None:
        self._cancel_show()
        self._cancel_hide()
        self._hide_tip()

    # Public aliases for show_tip / hide_tip.
    def show_tip(self, event=None) -> None:
        self._on_enter(event)

    def hide_tip(self, event=None) -> None:
        self._on_leave(event)
