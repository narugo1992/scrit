import hashlib
import os
import re
from typing import List, Optional
from urllib.parse import urlsplit

from ..errors import ResourceGone, UnexpectedResponse
from ..http import Fetcher, DownloadInfo, safe_name, filename_from_headers, ensure_extension


def host_of(url: str) -> str:
    host = (urlsplit(url).hostname or '').lower()
    return host[4:] if host.startswith('www.') else host


def segments(url: str) -> List[str]:
    return [seg for seg in urlsplit(url).path.split('/') if seg]


def short_hash(text: str, size: int = 16) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:size]


def clean_id(text: str) -> str:
    return re.sub(r'[^\w-]+', '_', text).strip('_')


def fetch_file(fx: Fetcher, url: str, out_dir: str, name: Optional[str] = None, *,
               gone_statuses=(404, 410), gone_if_redirected_to=(), **kwargs) -> str:
    """Download ``url`` into ``out_dir``; the final name comes from ``name``, the headers, or the URL."""
    part = os.path.join(out_dir, f'.part_{short_hash(url, 12)}')
    try:
        info: DownloadInfo = fx.download(url, part, **kwargs)
    except UnexpectedResponse as err:
        if err.status in gone_statuses:
            raise ResourceGone(f'{url!r} -> HTTP {err.status}') from err
        raise
    if any(marker in info.final_url for marker in gone_if_redirected_to):
        os.remove(part)
        raise ResourceGone(f'{url!r} was redirected to {info.final_url!r}')
    final = safe_name(name or filename_from_headers(info.headers) or os.path.basename(urlsplit(url).path))
    final = ensure_extension(final, info.content_type)
    target = os.path.join(out_dir, final)
    stem, ext = os.path.splitext(final)
    counter = 1
    while os.path.exists(target):
        counter += 1
        target = os.path.join(out_dir, f'{stem}_{counter}{ext}')
    os.replace(part, target)
    return target
