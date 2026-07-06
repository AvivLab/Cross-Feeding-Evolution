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


def create_labeled_entry(parent, label_text, default_value, row, tooltip_text=""):
    """
    Create a labeled entry field with optional tooltip.
    
    Parameters:
    -----------
    parent : tk.Widget
        Parent frame to place the label and entry
    label_text : str
        Text for the label
    default_value : str, int, float
        Default value to populate entry with
    row : int
        Grid row position
    tooltip_text : str, optional
        Tooltip text (optional; unused when the field has no tooltip widget)
    
    Returns:
    --------
    tuple : (label_widget, entry_widget)
    """
    lbl = tk.Label(parent, text=label_text, font=('Arial', 11))
    lbl.grid(row=row, column=0, sticky="w", padx=2, pady=1)
    
    ent = tk.Entry(parent, width=10)
    ent.grid(row=row, column=1, sticky="w", padx=2, pady=1)
    
    if default_value is None:
        default_value = ""
    elif not isinstance(default_value, (str, int, float)):
        default_value = str(default_value)
    
    ent.insert(0, default_value)
    return lbl, ent


def toggle_widgets(show_widgets, hide_widgets):
    """
    Show/hide widgets by toggling their grid visibility.
    
    Parameters:
    -----------
    show_widgets : list
        Widgets to show (call .grid())
    hide_widgets : list
        Widgets to hide (call .grid_remove())
    """
    for w in show_widgets:
        if w is not None:
            w.grid()
    for w in hide_widgets:
        if w is not None:
            w.grid_remove()


def show_initial_message(fig, canvas, starting_text):
    """
    Display a centered text message on a matplotlib figure.
    
    Useful for showing "Click Run to start" messages before simulation.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to display message on
    canvas : FigureCanvasTkAgg
        Canvas to redraw
    starting_text : str
        Message to display
    """
    fig.clf()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, starting_text, fontsize=10, ha='center', va='center')
    ax.axis('off')
    canvas.draw()


def create_enzyme_range_manager(parent_frame, enzyme_name, start_row):
    """
    Factory function to create enzyme range input managers.
    
    Allows users to specify multiple ranges for enzyme initialization,
    enabling multi-modal initial distributions.
    
    Parameters:
    -----------
    parent_frame : tk.Widget
        Parent widget to place the range manager
    enzyme_name : str
        Name of enzyme (e.g., "A", "B", "T")
    start_row : int
        Grid row to start at
    
    Returns:
    --------
    tuple : (add_field_func, get_ranges_func, entries_list, frame_widget)
        - add_field_func: Function to add a new range
        - get_ranges_func: Function to retrieve all ranges as list of tuples
        - entries_list: List of entry widgets
        - frame_widget: The LabelFrame container
    """
    frame = tk.LabelFrame(parent_frame, text=f"Enzyme {enzyme_name} Ranges", padx=5, pady=5)
    frame.grid(row=start_row, column=0, rowspan=6, sticky="nw", padx=0)
    entries = []
    
    def add_field(start_val="", end_val=""):
        row = len(entries)
        start_var = tk.StringVar(value=start_val)
        end_var = tk.StringVar(value=end_val)
        start_entry = tk.Entry(frame, width=6, textvariable=start_var)
        end_entry = tk.Entry(frame, width=6, textvariable=end_var)
        dash_label = tk.Label(frame, text="—")
        remove_btn = tk.Button(frame, text="x", command=lambda: remove_field(row), width=1)
        
        start_entry.grid(row=row, column=0, padx=(0,2))
        dash_label.grid(row=row, column=1)
        end_entry.grid(row=row, column=2, padx=(2,4))
        remove_btn.grid(row=row, column=3, padx=(2,2))
        
        entries.append((start_var, end_var, start_entry, end_entry, dash_label, remove_btn))
        refresh_rows()
    
    def remove_field(index):
        if index < len(entries) and entries[index] is not None:
            for widget in entries[index][2:]:
                widget.grid_forget()
            entries[index] = None
            refresh_rows()
    
    def refresh_rows():
        valid = [r for r in entries if r is not None]
        for i, row in enumerate(valid):
            start_var, end_var, start_entry, end_entry, dash_label, remove_btn = row
            start_entry.grid(row=i, column=0, padx=(0,2))
            dash_label.grid(row=i, column=1)
            end_entry.grid(row=i, column=2, padx=(2,4))
            remove_btn.grid(row=i, column=3, padx=(2,2))
        entries[:] = valid
    
    def get_ranges():
        ranges = []
        for r in entries:
            if r is not None:
                start, end = r[0].get(), r[1].get()
                try:
                    ranges.append((float(start), float(end)))
                except ValueError:
                    continue
        return ranges if ranges else None
    
    tk.Button(frame, text="+ Add Range", command=lambda: add_field()).grid(
        row=100, column=0, columnspan=4, sticky="w", pady=(3, 0))
    
    return add_field, get_ranges, entries, frame


def create_enzyme_range_manager_compact(parent_frame, enzyme_name, row_position):
    """
    Compact version of enzyme range manager for sweep GUIs.
    
    Similar to create_enzyme_range_manager but with simpler layout
    for use in parameter sweep interfaces.
    
    Parameters:
    -----------
    parent_frame : tk.Widget
        Parent widget
    enzyme_name : str
        Name of enzyme (e.g., "A", "B", "T")
    row_position : int
        Grid row position
    
    Returns:
    --------
    tuple : (add_field_func, get_ranges_func, entries_list)
    """
    frame = tk.LabelFrame(parent_frame, text=f"Enzyme {enzyme_name}", padx=3, pady=3)
    frame.grid(row=row_position, column=0, sticky="nw", padx=2)
    entries = []
    
    def add_field(start_val="0.0", end_val="1.0"):
        row = len(entries)
        start_var = tk.StringVar(value=start_val)
        end_var = tk.StringVar(value=end_val)
        start_entry = tk.Entry(frame, width=6, textvariable=start_var)
        end_entry = tk.Entry(frame, width=6, textvariable=end_var)
        dash_label = tk.Label(frame, text="—")
        remove_btn = tk.Button(frame, text="x", command=lambda: remove_field(row), width=1)
        
        start_entry.grid(row=row, column=0, padx=(0,2))
        dash_label.grid(row=row, column=1)
        end_entry.grid(row=row, column=2, padx=(2,4))
        remove_btn.grid(row=row, column=3, padx=(2,2))
        
        entries.append((start_var, end_var, start_entry, end_entry, dash_label, remove_btn))
    
    def remove_field(index):
        if index < len(entries) and entries[index] is not None:
            for widget in entries[index][2:]:
                widget.grid_forget()
            entries[index] = None
            refresh_rows()
    
    def refresh_rows():
        valid = [r for r in entries if r is not None]
        for i, row in enumerate(valid):
            start_var, end_var, start_entry, end_entry, dash_label, remove_btn = row
            start_entry.grid(row=i, column=0, padx=(0,2))
            dash_label.grid(row=i, column=1)
            end_entry.grid(row=i, column=2, padx=(2,4))
            remove_btn.grid(row=i, column=3, padx=(2,2))
        entries[:] = valid
    
    def get_ranges():
        ranges = []
        for r in entries:
            if r is not None:
                start, end = r[0].get(), r[1].get()
                try:
                    ranges.append((float(start), float(end)))
                except ValueError:
                    continue
        return ranges if ranges else None
    
    tk.Button(frame, text="+", command=lambda: add_field(), width=2).grid(row=100, column=0, columnspan=4, pady=2)
    
    return add_field, get_ranges, entries
