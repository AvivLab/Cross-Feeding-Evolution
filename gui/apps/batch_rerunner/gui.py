"""
Batch Re-Runner GUI: re-run hits or non-hits from a finished Batch Runner campaign.

Loads a campaign folder, re-runs each selected parameter set
with fresh random seeds, and reports how often the metric filters pass again.
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

from gui.apps.batch_rerunner.session_panel import RerunnerSessionPanel
from gui.common.widgets import CreateToolTip
from gui.common.tooltips import BATCH_RERUNNER_TOOLTIPS
from gui.models.registry import OptimizationModelSpec
from gui.persistence.json_io import make_read_json_maybe_gz_fn
from headless.hpc_common import is_primary_batch_session_folder
from headless.primary_events_chart import infer_session_id
from headless.primary_hit_rescreen import (
    DEFAULT_N_SEEDS,
    DEFAULT_RESCREEN_BASE_SEED,
    SCREEN_MODE_HITS,
    SCREEN_MODE_NON_HITS,
    collect_hit_specs,
    count_hit_specs,
    load_session_rescreen_context,
    rescreen_session,
)

_read_json_maybe_gz = make_read_json_maybe_gz_fn(plain_twin_fallback=False)


def _default_workers() -> int:
    try:
        return max(1, int(os.cpu_count() or 1))
    except Exception:
        return 1


def _resolve_session_from_path(path: str) -> Tuple[str, Optional[str]]:
    """Return ``(session_dir, session_id)`` from a folder or campaign JSON path."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        sid = infer_session_id(path) or ""
        return path, sid or None
    if os.path.isfile(path):
        session_dir = os.path.dirname(path)
        try:
            data = _read_json_maybe_gz(path)
        except Exception:
            data = None
        sid = ""
        if isinstance(data, dict):
            sid = str(data.get("full_save_session_id") or "").strip()
        if not sid:
            sid = infer_session_id(session_dir) or ""
        return session_dir, sid or None
    raise FileNotFoundError(f"Path not found: {path}")


def parse_batch_rerunner_run_options(
    *,
    session_path: str,
    n_seeds: str,
    max_non_hits: str,
    workers: str,
    screen_mode: str,
    limit_non_hits: bool,
    dedupe: bool,
    quiet_terminal: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate Batch Re-Runner GUI inputs (testable without Tk)."""
    raw = session_path.strip()
    if not raw:
        return None, "Select a Batch Runner session folder."
    try:
        session_dir, sid = _resolve_session_from_path(raw)
        if not is_primary_batch_session_folder(session_dir):
            return None, "Folder does not look like a Batch Runner session."
        n_seeds_i = int(n_seeds.strip())
        max_non_hits_i = int(max_non_hits.strip())
        workers_i = int(workers.strip())
    except ValueError as exc:
        return None, f"Invalid numeric input: {exc}"
    if n_seeds_i < 1 or n_seeds_i > 10_000:
        return None, "Seeds per point must be in [1, 10000]."
    if workers_i < 1:
        return None, "Workers must be >= 1."
    mode = screen_mode.strip()
    if mode not in ("hits", "non_hits", "both"):
        return None, "Choose hits, non-hits, or both."
    if mode in ("non_hits", "both") and limit_non_hits:
        if max_non_hits_i < 1:
            return None, "Max points must be >= 1 when limiting non-hits."
    elif mode in ("non_hits", "both") and not limit_non_hits:
        max_non_hits_i = 0
    else:
        max_non_hits_i = 0
    return {
        "session_dir": session_dir,
        "session_id": sid,
        "n_seeds": n_seeds_i,
        "max_non_hits": max_non_hits_i,
        "workers": workers_i,
        "dedupe": bool(dedupe),
        "quiet_terminal": bool(quiet_terminal),
        "screen_mode": mode,
    }, None


def batch_rerunner_gui(
    win: tk.Toplevel | tk.Tk,
    root: tk.Tk,
    model_spec: OptimizationModelSpec,
) -> None:
    if getattr(model_spec, "key", "") != "simulation":
        messagebox.showerror("Model", "Batch Re-Runner is only available for the Simulation model.")
        return

    win.title(f"Batch Re-Runner — {getattr(model_spec, 'label', 'Simulation')}")
    win.geometry("1180x820")

    session_path_var = tk.StringVar(value="")
    session_id_var = tk.StringVar(value="")
    n_hits_var = tk.StringVar(value="—")
    n_non_hits_var = tk.StringVar(value="—")
    status_var = tk.StringVar(value="Load a finished Batch Runner campaign to begin.")
    screen_mode_var = tk.StringVar(value="hits")
    n_seeds_var = tk.StringVar(value=str(DEFAULT_N_SEEDS))
    limit_non_hits_var = tk.BooleanVar(value=True)
    max_non_hits_var = tk.StringVar(value="500")
    workers_var = tk.StringVar(value=str(_default_workers()))
    dedupe_var = tk.BooleanVar(value=False)
    quiet_terminal_var = tk.BooleanVar(value=True)

    shell = ttk.Frame(win, padding=10)
    shell.pack(fill="both", expand=True)
    shell.grid_columnconfigure(0, weight=3)
    shell.grid_columnconfigure(1, weight=2)
    shell.grid_rowconfigure(1, weight=1)

    intro = ttk.Label(
        shell,
        text=(
            "Load a finished Batch Runner campaign, then re-run selected parameter sets "
            "with new random seeds. This checks whether hits were stable or depended on luck. "
            "Results are saved in the campaign folder: a summary of how often each set "
            "passes your metric filters again."
        ),
        wraplength=1120,
        justify="left",
    )
    intro.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

    left_col = ttk.Frame(shell)
    left_col.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

    session_panel = RerunnerSessionPanel(shell)
    session_panel.outer.grid(row=1, column=1, sticky="nsew")

    load_frame = ttk.LabelFrame(left_col, text="Load campaign", padding=8)
    load_frame.pack(fill="x", pady=(0, 8))

    path_row = ttk.Frame(load_frame)
    path_row.pack(fill="x", pady=2)
    path_label = ttk.Label(path_row, text="Session folder:")
    path_label.pack(side="left")
    path_entry = ttk.Entry(path_row, textvariable=session_path_var)
    path_entry.pack(side="left", fill="x", expand=True, padx=6)
    CreateToolTip(path_label, BATCH_RERUNNER_TOOLTIPS["Session folder"])
    CreateToolTip(path_entry, BATCH_RERUNNER_TOOLTIPS["Session folder"])

    def _refresh_preview() -> None:
        raw = session_path_var.get().strip()
        if not raw:
            session_id_var.set("")
            n_hits_var.set("—")
            n_non_hits_var.set("—")
            status_var.set("Load a finished Batch Runner campaign to begin.")
            session_panel.clear()
            return
        try:
            session_dir, sid = _resolve_session_from_path(raw)
            if not is_primary_batch_session_folder(session_dir):
                raise ValueError("Folder does not look like a Batch Runner session.")
            ctx = load_session_rescreen_context(session_dir, sid)
            session_path_var.set(session_dir)
            session_id_var.set(ctx.session_id)
            session_panel.set_context(ctx)
            hit_total, hit_unique = count_hit_specs(
                ctx.session_dir,
                ctx.session_id,
                screen_mode=SCREEN_MODE_HITS,
            )
            non_hit_total, non_hit_unique = count_hit_specs(
                ctx.session_dir,
                ctx.session_id,
                screen_mode=SCREEN_MODE_NON_HITS,
            )
            n_hits_var.set(f"{hit_total} ({hit_unique} unique)")
            n_non_hits_var.set(f"{non_hit_total} ({non_hit_unique} unique)")
            status_var.set(
                f"Loaded campaign {ctx.session_id}: "
                f"{hit_total} hits ({hit_unique} unique), "
                f"{non_hit_total} non-hits ({non_hit_unique} unique)."
            )
        except Exception as exc:
            session_id_var.set("")
            n_hits_var.set("—")
            n_non_hits_var.set("—")
            session_panel.clear()
            status_var.set(f"Could not load session: {exc}")

    def _choose_folder() -> None:
        sel = filedialog.askdirectory(
            title="Select Batch Runner session folder",
            parent=win,
        )
        if sel:
            session_path_var.set(sel)
            _refresh_preview()

    browse_btn = ttk.Button(path_row, text="Choose Session Folder", command=_choose_folder)
    browse_btn.pack(side="left", padx=(0, 4))

    info_row = ttk.Frame(load_frame)
    info_row.pack(fill="x", pady=(6, 0))
    info_row.grid_columnconfigure(1, weight=1)
    for row, (label_text, var) in enumerate(
        (
            ("Session id:", session_id_var),
            ("Hits:", n_hits_var),
            ("Non-hits:", n_non_hits_var),
        )
    ):
        ttk.Label(info_row, text=label_text).grid(row=row, column=0, sticky="nw", pady=1)
        ttk.Label(info_row, textvariable=var, anchor="w", justify="left").grid(
            row=row,
            column=1,
            sticky="ew",
            padx=(8, 0),
            pady=1,
        )

    opts = ttk.LabelFrame(left_col, text="Re-screen options", padding=8)
    opts.pack(fill="x", pady=(0, 8))

    mode_row = ttk.Frame(opts)
    mode_row.pack(fill="x", pady=2)
    ttk.Label(mode_row, text="Re-screen:").pack(side="left")
    for val, label in (
        ("hits", "Hits Only"),
        ("non_hits", "Non-Hits Only"),
        ("both", "Both (Hits Then Non-Hits)"),
    ):
        rb = ttk.Radiobutton(mode_row, text=label, value=val, variable=screen_mode_var)
        rb.pack(side="left", padx=(8, 0))

    row1 = ttk.Frame(opts)
    row1.pack(fill="x", pady=4)
    seeds_label = ttk.Label(row1, text="Seeds per point (N):")
    seeds_label.pack(side="left")
    seeds_entry = ttk.Entry(row1, textvariable=n_seeds_var, width=8)
    seeds_entry.pack(side="left", padx=6)
    CreateToolTip(seeds_label, BATCH_RERUNNER_TOOLTIPS["N seeds"])
    CreateToolTip(seeds_entry, BATCH_RERUNNER_TOOLTIPS["N seeds"])

    workers_label = ttk.Label(row1, text="Workers:")
    workers_label.pack(side="left", padx=(16, 0))
    workers_entry = ttk.Entry(row1, textvariable=workers_var, width=8)
    workers_entry.pack(side="left", padx=6)
    CreateToolTip(workers_label, BATCH_RERUNNER_TOOLTIPS["Workers"])
    CreateToolTip(workers_entry, BATCH_RERUNNER_TOOLTIPS["Workers"])

    non_hits_row = ttk.Frame(opts)
    non_hits_row.pack(fill="x", pady=(4, 0))
    limit_non_hits_cb = ttk.Checkbutton(
        non_hits_row,
        text="Don't run all non-hits",
        variable=limit_non_hits_var,
    )
    limit_non_hits_cb.pack(side="left")
    CreateToolTip(limit_non_hits_cb, BATCH_RERUNNER_TOOLTIPS["Limit non-hits"])
    max_non_hits_label = ttk.Label(non_hits_row, text="Max points:")
    max_non_hits_label.pack(side="left", padx=(16, 0))
    max_non_hits_entry = ttk.Entry(non_hits_row, textvariable=max_non_hits_var, width=8)
    max_non_hits_entry.pack(side="left", padx=6)
    CreateToolTip(max_non_hits_label, BATCH_RERUNNER_TOOLTIPS["Max non-hits"])
    CreateToolTip(max_non_hits_entry, BATCH_RERUNNER_TOOLTIPS["Max non-hits"])

    dedupe_cb = ttk.Checkbutton(opts, text="Dedupe identical parameter vectors", variable=dedupe_var)
    dedupe_cb.pack(anchor="w", pady=(4, 0))
    CreateToolTip(dedupe_cb, BATCH_RERUNNER_TOOLTIPS["Dedupe"])

    quiet_terminal_cb = ttk.Checkbutton(
        opts,
        text="Quiet terminal output",
        variable=quiet_terminal_var,
    )
    quiet_terminal_cb.pack(anchor="w", pady=(4, 0))
    CreateToolTip(quiet_terminal_cb, BATCH_RERUNNER_TOOLTIPS["Quiet terminal"])

    def _update_non_hits_limit_ui(*_args: object) -> None:
        mode = screen_mode_var.get().strip()
        applies = mode in ("non_hits", "both")
        if applies:
            non_hits_row.pack(fill="x", pady=(4, 0), before=dedupe_cb)
            limit_non_hits_cb.configure(state="normal")
            limit_on = bool(limit_non_hits_var.get())
            entry_state = "normal" if limit_on else "disabled"
            max_non_hits_entry.configure(state=entry_state)
        else:
            non_hits_row.pack_forget()

    screen_mode_var.trace_add("write", _update_non_hits_limit_ui)
    limit_non_hits_var.trace_add("write", _update_non_hits_limit_ui)
    _update_non_hits_limit_ui()

    run_row = ttk.Frame(left_col)
    run_row.pack(fill="x", pady=(0, 6))
    prog = ttk.Progressbar(run_row, mode="determinate", maximum=100)
    prog.pack(fill="x", pady=(0, 6))

    run_btn = ttk.Button(run_row, text="Run Re-Screen", width=18)
    run_btn.pack(side="left")

    rescreen_running = {"value": False}

    def _go_back() -> None:
        if rescreen_running["value"]:
            if not messagebox.askyesno(
                "Re-screen running",
                "A re-screen is still running. Close anyway?\n"
                "(The background job may continue until it finishes.)",
                parent=win,
            ):
                return
        try:
            prog.configure(value=0)
        except tk.TclError:
            pass
        try:
            root.deiconify()
        except tk.TclError:
            pass
        win.destroy()

    back_btn = ttk.Button(run_row, text="← Back", command=_go_back, width=10)
    back_btn.pack(side="left", padx=(8, 0))

    win.protocol("WM_DELETE_WINDOW", _go_back)

    status_label = ttk.Label(left_col, textvariable=status_var, justify="left")
    status_label.pack(fill="x", pady=(4, 0))

    def _update_wraplengths(event: Optional[tk.Event] = None) -> None:
        try:
            shell_width = int(shell.winfo_width())
            left_width = int(left_col.winfo_width())
        except tk.TclError:
            return
        if shell_width > 1:
            intro.configure(wraplength=max(200, shell_width - 24))
        if left_width > 1:
            status_label.configure(wraplength=max(200, left_width - 16))

    shell.bind("<Configure>", _update_wraplengths)
    left_col.bind("<Configure>", _update_wraplengths)

    def _parse_run_options() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        return parse_batch_rerunner_run_options(
            session_path=session_path_var.get(),
            n_seeds=n_seeds_var.get(),
            max_non_hits=max_non_hits_var.get(),
            workers=workers_var.get(),
            screen_mode=screen_mode_var.get(),
            limit_non_hits=bool(limit_non_hits_var.get()),
            dedupe=bool(dedupe_var.get()),
            quiet_terminal=bool(quiet_terminal_var.get()),
        )

    def _run() -> None:
        opts_dict, err = _parse_run_options()
        if err is not None or opts_dict is None:
            messagebox.showerror("Invalid input", err or "Unknown error.", parent=win)
            return

        mode = opts_dict["screen_mode"]
        if mode in ("non_hits", "both") and int(opts_dict["max_non_hits"]) <= 0:
            if not messagebox.askyesno(
                "Run all non-hits",
                "You chose to re-screen every unique non-hit in the campaign "
                "(often very large). Continue anyway?",
                parent=win,
            ):
                return

        run_btn.configure(state="disabled")
        rescreen_running["value"] = True
        prog.configure(mode="determinate", maximum=100, value=0)
        q: queue.Queue = queue.Queue()

        def worker() -> None:
            modes: List[str] = []
            if mode == "both":
                modes = [SCREEN_MODE_HITS, SCREEN_MODE_NON_HITS]
            elif mode == "non_hits":
                modes = [SCREEN_MODE_NON_HITS]
            else:
                modes = [SCREEN_MODE_HITS]

            summaries: List[Dict[str, Any]] = []
            try:
                mode_point_counts: List[Tuple[str, int]] = []
                for sm in modes:
                    # max_hits=0 means no cap (re-screen every hit; non-hits use max_non_hits when set).
                    max_hits = 0 if sm == SCREEN_MODE_HITS else int(opts_dict["max_non_hits"])
                    n_points = len(
                        collect_hit_specs(
                            opts_dict["session_dir"],
                            opts_dict["session_id"],
                            max_hits=max_hits,
                            dedupe_params=bool(opts_dict["dedupe"]),
                            screen_mode=sm,
                        )
                    )
                    label = "hits" if sm == SCREEN_MODE_HITS else "non-hits"
                    mode_point_counts.append((label, n_points))
                n_seeds = int(opts_dict["n_seeds"])
                total_trials = sum(n * n_seeds for _, n in mode_point_counts)
                progress_total = max(1, total_trials)
                q.put(
                    (
                        "progress",
                        {
                            "done": 0,
                            "total": progress_total,
                            "message": (
                                f"Preparing re-screen of {total_trials} re-runs "
                                f"({', '.join(f'{n} {label}' for label, n in mode_point_counts)}, "
                                f"{n_seeds} seeds each)…"
                            ),
                        },
                    )
                )

                trial_offset = 0
                for i, sm in enumerate(modes):
                    label = mode_point_counts[i][0]
                    mode_points = mode_point_counts[i][1]
                    max_hits = 0 if sm == SCREEN_MODE_HITS else int(opts_dict["max_non_hits"])
                    offset = trial_offset
                    mode_trials = mode_points * n_seeds

                    # Fold per-mode trial progress into one overall progress bar for the GUI.
                    def _progress(
                        done: int,
                        _total: int,
                        message: str,
                        *,
                        _label: str = label,
                        _offset: int = offset,
                        _total_all: int = progress_total,
                        _mode_i: int = i,
                        _n_modes: int = len(modes),
                    ) -> None:
                        overall_done = min(_total_all, _offset + done)
                        q.put(
                            (
                                "progress",
                                {
                                    "done": overall_done,
                                    "total": _total_all,
                                    "message": (
                                        f"[{_mode_i + 1}/{_n_modes}] {_label}: {message}"
                                    ),
                                },
                            )
                        )

                    summary = rescreen_session(
                        opts_dict["session_dir"],
                        session_id=opts_dict["session_id"],
                        n_seeds=int(opts_dict["n_seeds"]),
                        rescreen_base_seed=DEFAULT_RESCREEN_BASE_SEED,
                        max_hits=max_hits,
                        dedupe_params=bool(opts_dict["dedupe"]),
                        screen_mode=sm,
                        workers=int(opts_dict["workers"]),
                        progress=not bool(opts_dict["quiet_terminal"]),
                        progress_callback=_progress,
                        write_rehit_heatmap=False,
                    )
                    summaries.append(summary)
                    trial_offset += mode_trials
                q.put(("done", summaries))
            except Exception as exc:
                q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        def poll() -> None:
            try:
                kind, payload = q.get_nowait()
            except queue.Empty:
                win.after(200, poll)
                return

            if kind == "progress":
                done = int(payload.get("done", 0))
                total = max(1, int(payload.get("total", 1)))
                prog.configure(mode="determinate", maximum=total, value=min(done, total))
                status_var.set(str(payload.get("message", "")))
                win.after(200, poll)
                return

            if kind == "status":
                status_var.set(str(payload))
                win.after(200, poll)
                return

            prog.configure(value=prog["maximum"])
            run_btn.configure(state="normal")
            rescreen_running["value"] = False

            if kind == "error":
                try:
                    prog.configure(value=0)
                except tk.TclError:
                    pass
                status_var.set(f"Re-screen failed: {payload}")
                messagebox.showerror("Re-screen failed", str(payload), parent=win)
                return

            summaries: List[Dict[str, Any]] = payload
            lines: List[str] = []
            for summary in summaries:
                sm = summary.get("screen_mode", "")
                label = "Hits" if sm == SCREEN_MODE_HITS else "Non-hits"
                n_screened = summary.get("n_hits_screened", "?")
                mean_rate = summary.get("mean_hit_rate")
                csv_path = summary.get("csv_path", "")
                lines.append(f"{label}: screened {n_screened} points")
                if mean_rate is not None:
                    try:
                        lines.append(f"  mean re-hit rate = {float(mean_rate):.3f}")
                    except (TypeError, ValueError):
                        lines.append(f"  mean re-hit rate = {mean_rate}")
                if csv_path:
                    lines.append(f"  saved to {csv_path}")
                hit_counts_csv = summary.get("hit_counts_csv", "")
                if hit_counts_csv:
                    lines.append(f"  updated batch CSV {hit_counts_csv}")
            status_var.set("Re-screen complete.\n" + "\n".join(lines))
            _refresh_preview()
            messagebox.showinfo(
                "Re-screen complete",
                "\n".join(lines) or "Finished.",
                parent=win,
            )

        win.after(200, poll)

    run_btn.configure(command=_run)
