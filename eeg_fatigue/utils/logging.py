import io
import sys


class _Logger:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_logging(log_path: str) -> tuple[_Logger, object]:
    """Set up dual logging to file and stdout. Returns (logger, original_stdout)."""
    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    try:
        safe_stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        safe_stdout = original_stdout
    logger = _Logger(safe_stdout, log_file)
    return logger, original_stdout
