"""File-descriptor-level stdout/stderr capture for C-level prints.

pybind11's ostream_redirect does not capture C printf output; redirecting the
process-level file descriptors 1/2 around simulation calls is the only way to
silence the noisy CUDA debug prints. By default only lines containing the
configured keywords are kept; ``verbose`` passes everything through.

On Windows, only the FIRST fd-level redirection in a process is honored by the
C runtime (the CRT stdout/stderr FILE* objects cache the handle resolved at
first use; later ``os.dup2`` calls are ignored by C ``printf``). The
``RedirectRouter`` therefore installs one persistent capture at first use and
switches the per-case log file via ``begin_case``/``end_case``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_KEYWORDS = (b"Error", b"PCG", b"ERROR")


class RedirectRouter:
    """Persistent fd-level capture with a switchable per-case log file.

    Install once per process (first ``begin_case``) and call ``begin_case`` /
    ``end_case`` around each simulation case. ``stop`` restores the original
    descriptors; call it at process/session teardown.
    """

    def __init__(
        self,
        keywords: tuple[bytes | str, ...] = DEFAULT_KEYWORDS,
        verbose: bool = False,
    ) -> None:
        self.keywords = tuple(k.encode() if isinstance(k, str) else k for k in keywords)
        self.verbose = verbose
        self._saved_out: int | None = None
        self._saved_err: int | None = None
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._active = False
        self._current_log = None

    def start(self) -> None:
        """Install the persistent fd redirection (idempotent)."""
        if self._active:
            return
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, 1)
        os.dup2(self._write_fd, 2)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._active = True

    def begin_case(self, log_path: str | Path) -> None:
        """Start routing captured output to a per-case log file."""
        self.start()
        self.end_case()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._current_log = open(log_path, "wb")

    def end_case(self) -> None:
        """Stop routing to the current per-case log (idempotent)."""
        if self._current_log is not None:
            self._current_log.close()
            self._current_log = None

    def stop(self) -> None:
        """Restore original descriptors and stop the reader thread."""
        if not self._active:
            return
        self.end_case()
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        if self._write_fd is not None:
            os.close(self._write_fd)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._read_fd is not None:
            os.close(self._read_fd)
        if self._saved_out is not None:
            os.close(self._saved_out)
        if self._saved_err is not None:
            os.close(self._saved_err)
        self._active = False

    def _reader(self) -> None:
        pending = b""
        while True:
            chunk = os.read(self._read_fd, 4096)
            if not chunk:
                break
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                self._emit(line + b"\n")
        if pending:
            self._emit(pending)

    def _emit(self, line: bytes) -> None:
        if self.verbose or any(keyword in line for keyword in self.keywords):
            if self._current_log is not None:
                self._current_log.write(line)
                self._current_log.flush()


class FdOutputRedirect:
    """Context manager that captures fd 1/2 into a per-case log file.

    Note: on Windows only the first use per process is reliable; prefer
    ``RedirectRouter`` inside the test suite.
    """

    def __init__(
        self,
        log_path: str | Path,
        keywords: tuple[bytes | str, ...] = DEFAULT_KEYWORDS,
        verbose: bool = False,
    ) -> None:
        self.log_path = Path(log_path)
        self.keywords = tuple(k.encode() if isinstance(k, str) else k for k in keywords)
        self.verbose = verbose
        self._saved_out: int | None = None
        self._saved_err: int | None = None
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "FdOutputRedirect":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, 1)
        os.dup2(self._write_fd, 2)
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        if self._write_fd is not None:
            os.close(self._write_fd)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._read_fd is not None:
            os.close(self._read_fd)
        if self._saved_out is not None:
            os.close(self._saved_out)
        if self._saved_err is not None:
            os.close(self._saved_err)

    def _reader(self) -> None:
        """Read captured output, filtering noise unless verbose."""
        with open(self.log_path, "wb") as log:
            pending = b""
            while True:
                chunk = os.read(self._read_fd, 4096)
                if not chunk:
                    break
                pending += chunk
                lines = pending.split(b"\n")
                pending = lines.pop()
                for line in lines:
                    self._emit(log, line + b"\n")
            if pending:
                self._emit(log, pending)

    def _emit(self, log, line: bytes) -> None:
        if self.verbose or any(keyword in line for keyword in self.keywords):
            log.write(line)
            log.flush()
