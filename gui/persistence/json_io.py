"""Shared JSON I/O helpers for plain and gzipped files."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from typing import Any, Callable, Mapping


def read_json_maybe_gz(
    path: str,
    *,
    plain_twin_fallback: bool = False,
    fast_json_module=None,
) -> dict:
    """Read JSON from `.json` or `.json.gz`, with optional plain-json fallback."""
    if path.endswith(".gz"):
        try:
            if fast_json_module is not None:
                with gzip.open(path, "rb") as f:
                    return fast_json_module.loads(f.read())
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            if plain_twin_fallback:
                twin = path[:-3]
                if os.path.exists(twin):
                    with open(twin, "r") as f:
                        return json.load(f)
            raise
    with open(path, "r") as f:
        return json.load(f)


def write_json_maybe_gz_atomic(
    path: str,
    payload: Mapping[str, Any],
    *,
    indent: int | None = 2,
    gzip_compresslevel: int | None = None,
    ensure_ascii: bool = True,
    fast_json_module=None,
) -> None:
    """Atomically write JSON payload to `.json` or `.json.gz` path."""
    target_dir = os.path.dirname(path) or "."
    base = os.path.basename(path)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{base}.tmp.", dir=target_dir)
    os.close(fd)
    try:
        if path.endswith(".gz"):
            if gzip_compresslevel is None:
                level = None
            else:
                level = int(max(1, min(9, int(gzip_compresslevel))))
            if fast_json_module is not None:
                raw = fast_json_module.dumps(payload, ensure_ascii=ensure_ascii).encode("utf-8")
                kwargs = {"compresslevel": level} if level is not None else {}
                with gzip.open(tmp_path, "wb", **kwargs) as f:
                    f.write(raw)
            else:
                kwargs = {"compresslevel": level} if level is not None else {}
                with gzip.open(tmp_path, "wt", encoding="utf-8", **kwargs) as f:
                    json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii)
        else:
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def make_read_json_maybe_gz_fn(
    *,
    plain_twin_fallback: bool = False,
    fast_json_module=None,
) -> Callable[[str], dict]:
    """Return a reader closure with fixed ``plain_twin_fallback`` / optional fast JSON."""

    def read_json(path: str) -> dict:
        return read_json_maybe_gz(
            path,
            plain_twin_fallback=plain_twin_fallback,
            fast_json_module=fast_json_module,
        )

    return read_json


def make_write_json_maybe_gz_atomic_fn(
    *,
    indent: int | None = 2,
    default_gzip_compresslevel: int | None = None,
    ensure_ascii: bool = True,
    fast_json_module=None,
) -> Callable[..., None]:
    """Return a writer closure; omit ``gzip_compresslevel`` to use ``default_gzip_compresslevel``."""

    def write_json(
        path: str,
        payload: Mapping[str, Any],
        *,
        gzip_compresslevel: int | None = None,
    ) -> None:
        level = default_gzip_compresslevel if gzip_compresslevel is None else gzip_compresslevel
        write_json_maybe_gz_atomic(
            path,
            payload,
            indent=indent,
            gzip_compresslevel=level,
            ensure_ascii=ensure_ascii,
            fast_json_module=fast_json_module,
        )

    return write_json

