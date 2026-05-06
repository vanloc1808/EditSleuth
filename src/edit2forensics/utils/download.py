"""Lightweight image-download utility with deterministic caching.

Used by adapters when a source dataset provides URLs instead of local files
(Pico-Banana's `open_image_input_url` is the motivating case).

Design
------
* Cache filenames are derived from a hash of the URL, so the same URL
  always maps to the same path — safe to call repeatedly.
* Failures are non-fatal: return `None` and let the adapter decide whether
  to skip the record or raise.
* No retries here. If the research run needs robust downloads, wrap this
  in a caller-side retry policy; keeping this function pure makes testing
  easier.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


def url_to_cache_path(url: str, cache_dir: Path) -> Path:
    """Map a URL to a deterministic cache path.

    The file extension is preserved from the URL when available,
    else defaults to ``.jpg`` (Pico-Banana's Flickr URLs carry a
    proper extension).
    """
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    return cache_dir / f"{h}{ext}"


def download_image(
    url: str,
    cache_dir: Path,
    timeout: float = 15.0,
    skip_if_exists: bool = True,
) -> Path | None:
    """Download `url` into `cache_dir` and return the local path.

    Returns ``None`` on any failure (network error, non-200, empty body).
    The function is idempotent: if the cache file already exists and
    `skip_if_exists` is True, no network call is made.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = url_to_cache_path(url, cache_dir)
    if skip_if_exists and dest.exists() and dest.stat().st_size > 0:
        return dest

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        if resp.status_code != 200:
            log.warning("download_image: HTTP %d for %s", resp.status_code, url)
            return None
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            log.warning("download_image: empty body for %s", url)
            return None
        tmp.rename(dest)
        return dest
    except requests.RequestException as e:
        log.warning("download_image: %s for %s", type(e).__name__, url)
        return None
