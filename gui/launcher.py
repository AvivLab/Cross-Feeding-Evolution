import os
import warnings
import tkinter as tk
import sys

# Suppress multiprocessing resource_tracker shutdown warnings.
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
warnings.filterwarnings("ignore", message=".*resource_tracker.*")
warnings.filterwarnings("ignore", message=".*leaked semaphore.*")

# Also propagate suppression to multiprocessing child processes (resource_tracker is a separate process).
_pw = os.environ.get("PYTHONWARNINGS", "")
_extra = "ignore::UserWarning:multiprocessing.resource_tracker"
if _extra not in _pw:
    os.environ["PYTHONWARNINGS"] = ",".join([p for p in [_pw, _extra] if p])

from gui.apps.individual.gui import individual_gui
from gui.apps.gradient_descent.gui import gradient_descent_gui
from gui.apps.neutral_comparison.gui import batch_runner_gui
from gui.apps.batch_rerunner.gui import batch_rerunner_gui
from gui.models.registry import get_model_by_key

# === GUI ===
def launcher():
    """Open the top-level launcher window and route to sub-GUIs."""
    root = tk.Tk()
    root.title("Simulation Launcher")

    # Button registry (add new launch targets here)
    def go_to_individual_gui():
        """Launch individual simulation GUI."""
        spec = get_model_by_key("simulation")
        root.withdraw()
        win = tk.Toplevel(root)
        individual_gui(win, root, model_spec=spec)
        try:
            win.title(f"Individual Simulation — {spec.label}")
        except Exception:
            pass

        def on_close():
            root.deiconify()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def go_to_gradient_descent_gui():
        """Launch gradient descent GUI."""
        spec = get_model_by_key("simulation")
        root.withdraw()
        win = tk.Toplevel(root)
        gradient_descent_gui(win, root, model_spec=spec)
        # Label the window with the chosen model
        try:
            win.title(f"Gradient Descent Optimization — {spec.label}")
        except Exception:
            pass

        def on_close():
            root.deiconify()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def go_to_batch_runner_gui():
        """Launch Batch Runner (simulation batch campaigns)."""
        spec = get_model_by_key("simulation")
        root.withdraw()
        win = tk.Toplevel(root)
        batch_runner_gui(win, root, model_spec=spec)

        def on_close():
            root.deiconify()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    def go_to_batch_rerunner_gui():
        """Launch Batch Re-Runner (re-run hits or non-hits from finished campaigns)."""
        spec = get_model_by_key("simulation")
        root.withdraw()
        win = tk.Toplevel(root)
        batch_rerunner_gui(win, root, model_spec=spec)

        def on_close():
            root.deiconify()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

    buttons = [
        ("Individual Simulation GUI", go_to_individual_gui),
        ("Gradient Descent Optimization", go_to_gradient_descent_gui),
        ("Batch Runner", go_to_batch_runner_gui),
        ("Batch Re-Runner", go_to_batch_rerunner_gui),
        ("Quit", sys.exit),
    ]

    # Dynamically size the window based on number of buttons
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    max_width = int(screen_width * 0.25)
    # Estimate height: header + per-button height + margins
    header_h = 60
    per_button_h = 44  # approx button height + vertical padding
    margin_h = 30
    est_height = header_h + len(buttons) * per_button_h + margin_h
    # Cap to a reasonable fraction of screen
    max_height = min(est_height, int(screen_height * 0.5))

    x_offset = 0
    y_offset = 0
    root.geometry(f"{max_width}x{max_height}+{x_offset}+{y_offset}")

    tk.Label(root, text="Choose which GUI to launch:", font=("Helvetica", 12)).pack(pady=10)

    # Render buttons
    for text, cmd in buttons:
        tk.Button(root, text=text, command=cmd, width=25).pack(pady=5)

    root.mainloop()