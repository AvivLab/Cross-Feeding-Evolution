"""
Terminal-driven pause/resume for long-running GUI workers.

When the app is started from a terminal (stdin is a TTY), a daemon thread reads
lines from stdin. Typing ``pause`` / ``resume`` (or ``p`` / ``r``) dispatches to
all registered targets.

Immediate callbacks run on the stdin thread (for thread-safe primitives like
``threading.Event``). A follow-up ``root.after(0, ...)`` sync runs on the Tk main
thread for button labels and status text when the event loop is responsive.
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, List, Optional

try:
    import tkinter as tk
except ImportError:  # pragma: no cover
    tk = None  # type: ignore

_lock = threading.Lock()
_hooks: List["_Hook"] = []
_listener_started = False


class _Hook:
    __slots__ = (
        "root",
        "immediate_pause",
        "immediate_resume",
        "main_thread_after_pause",
        "main_thread_after_resume",
    )

    def __init__(
        self,
        root: "tk.Misc",
        immediate_pause: Callable[[], None],
        immediate_resume: Callable[[], None],
        main_thread_after_pause: Callable[[], None],
        main_thread_after_resume: Callable[[], None],
    ) -> None:
        self.root = root
        self.immediate_pause = immediate_pause
        self.immediate_resume = immediate_resume
        self.main_thread_after_pause = main_thread_after_pause
        self.main_thread_after_resume = main_thread_after_resume


def _stdin_loop() -> None:
    global _listener_started
    try:
        for line in sys.stdin:
            cmd = str(line or "").strip().lower()
            if not cmd or cmd.startswith("#"):
                continue
            if cmd in ("pause", "p"):
                _dispatch_pause()
            elif cmd in ("resume", "r", "unpause", "continue", "c"):
                _dispatch_resume()
            elif cmd in ("help", "h", "?"):
                sys.stderr.write(
                    "[terminal pause] commands: pause | resume | help (short: p, r, h)\n"
                )
                sys.stderr.flush()
    except Exception:
        pass
    finally:
        with _lock:
            _listener_started = False


def _ensure_listener() -> None:
    global _listener_started
    if not sys.stdin.isatty():
        return
    with _lock:
        if _listener_started:
            return
        _listener_started = True
    t = threading.Thread(target=_stdin_loop, name="terminal-pause-stdin", daemon=True)
    t.start()
    try:
        sys.stderr.write(
            "[terminal pause] listening on stdin — type 'pause' or 'resume' (see 'help')\n"
        )
        sys.stderr.flush()
    except Exception:
        pass


def _dispatch_pause() -> None:
    with _lock:
        hooks = list(_hooks)
    for h in hooks:
        try:
            h.immediate_pause()
        except Exception:
            pass
        try:
            h.root.after(0, h.main_thread_after_pause)
        except Exception:
            pass


def _dispatch_resume() -> None:
    with _lock:
        hooks = list(_hooks)
    for h in hooks:
        try:
            h.immediate_resume()
        except Exception:
            pass
        try:
            h.root.after(0, h.main_thread_after_resume)
        except Exception:
            pass


def register_terminal_pause_hooks(
    root: "tk.Misc",
    *,
    immediate_pause: Callable[[], None],
    immediate_resume: Callable[[], None],
    main_thread_after_pause: Callable[[], None],
    main_thread_after_resume: Callable[[], None],
) -> Callable[[], None]:
    """
    Register pause/resume hooks for the lifetime of a run.

    Returns ``unregister`` to call when the run finishes (also removes hooks if
    the Tk window is destroyed).
    """
    if tk is None or root is None:
        return lambda: None

    hook = _Hook(
        root,
        immediate_pause,
        immediate_resume,
        main_thread_after_pause,
        main_thread_after_resume,
    )

    def _unregister() -> None:
        with _lock:
            try:
                _hooks.remove(hook)
            except ValueError:
                pass

    def _on_destroy(evt=None) -> None:
        try:
            if evt is not None and getattr(evt, "widget", None) is not root:
                return
        except Exception:
            pass
        _unregister()

    with _lock:
        _hooks.append(hook)
    _ensure_listener()

    try:
        root.bind("<Destroy>", _on_destroy, add=True)
    except Exception:
        pass

    return _unregister
