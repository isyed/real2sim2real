"""Serve local review media on loopback with byte ranges for video seeking."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import mimetypes
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit


ALLOWED_SUFFIXES = {
    ".html",
    ".htm",
    ".css",
    ".js",
    ".json",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".webm",
}


def byte_range(value, size):
    """Return an inclusive single range; reject invalid or unsatisfiable requests."""
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value or size <= 0:
        raise ValueError("Unsupported byte range")
    start, end = value[6:].split("-", 1)
    if not start:
        length = int(end)
        if length <= 0:
            raise ValueError("Empty suffix range")
        return max(0, size - length), size - 1
    first = int(start)
    last = min(int(end), size - 1) if end else size - 1
    if first < 0 or first >= size or last < first:
        raise ValueError("Unsatisfiable byte range")
    return first, last


def resolve_media(root, url):
    root = Path(root).resolve()
    parts = Path(unquote(urlsplit(url).path).lstrip("/")).parts
    if any(part.startswith(".") for part in parts):
        raise ValueError("Hidden files and parent traversal are not served")
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Resource outside the review directory")
    if path.is_dir():
        path = (path / "review.html").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Index outside the review directory")
    if path.suffix.lower() not in ALLOWED_SUFFIXES or not path.is_file():
        raise FileNotFoundError("No review resource at this path")
    return path


def handler(root):
    class ReviewHandler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.serve(head_only=True)

        def do_GET(self):
            self.serve(head_only=False)

        def serve(self, *, head_only):
            try:
                path = resolve_media(root, self.path)
            except (ValueError, FileNotFoundError):
                self.send_error(404)
                return
            size = path.stat().st_size
            try:
                selected = byte_range(self.headers.get("Range"), size)
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            first, last = selected if selected is not None else (0, size - 1)
            self.send_response(206 if selected is not None else 200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(last - first + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if selected is not None:
                self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.end_headers()
            if head_only:
                return
            try:
                with path.open("rb") as stream:
                    stream.seek(first)
                    remaining = last - first + 1
                    while remaining:
                        data = stream.read(min(1024 * 1024, remaining))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browser seek/navigation can cancel an in-flight range.

    return ReviewHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seconds", type=float, default=28800)
    args = parser.parse_args()
    if (
        not args.directory.is_dir()
        or not 0 <= args.port <= 65535
        or not 0 < args.seconds <= 86400
    ):
        parser.error("Use a review directory, valid port and lifetime up to 24 hours")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.directory))
    timer = threading.Timer(args.seconds, server.shutdown)
    timer.daemon = True
    timer.start()
    print(
        f"Review server: http://127.0.0.1:{server.server_port}/ (expires after {args.seconds:g} seconds)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        timer.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
