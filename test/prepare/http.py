import ipaddress
import logging
import mimetypes
import os
import re
import socket
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlsplit, urljoin

import pyrfc6266
import requests

from .errors import ResourceTransient, UnexpectedResponse, UnsafeUrl, TooLarge

MODERN_UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
             'Chrome/129.0.0.0 Safari/537.36')

_RETRY_STATUS = {429, 500, 502, 503, 504}
_REDIRECT_STATUS = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 6
_HOST_TTL = 600.0

# one resource may not use more disk than this; a runner has about 14 GB free
MAX_RESOURCE_BYTES = 6 * 1024 ** 3
_CHUNK_SIZE = 1 << 20


@dataclass
class DownloadInfo:
    size: int
    content_type: str
    headers: Dict[str, str]
    final_url: str


def safe_name(name: str, default: str = 'file') -> str:
    """Make one path component safe to create on disk."""
    name = re.sub(r'[\x00-\x1f/\\]+', '_', name or '').strip().strip('.')
    return name or default


def _undo_latin1(name: str) -> str:
    """Servers like Google Drive put raw UTF-8 bytes in the header; requests decodes those as latin-1."""
    try:
        return name.encode('latin-1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def filename_from_headers(headers) -> Optional[str]:
    disposition = headers.get('Content-Disposition')
    if not disposition:
        return None
    name = pyrfc6266.parse_filename(disposition)
    return _undo_latin1(name) if name else None


def ensure_extension(name: str, content_type: str) -> str:
    if os.path.splitext(name)[1]:
        return name
    return name + (mimetypes.guess_extension(content_type.split(';')[0].strip()) or '')


class Fetcher:
    """HTTP helper shared by the site handlers: fixed modern UA, per-host pacing, bounded retries."""

    def __init__(self, gap: float = 1.0, gaps: Optional[Dict[str, float]] = None,
                 timeout=(15, 90), retries: int = 3, user_agent: str = MODERN_UA):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = user_agent
        self.gap = gap
        self.gaps = dict(gaps or {})
        self.timeout = timeout
        self.retries = retries
        self._last: Dict[str, float] = {}
        self._public_hosts: Dict[str, float] = {}
        self.resource_limit: Optional[int] = None
        self.resource_bytes = 0

    def begin_resource(self, limit: Optional[int] = MAX_RESOURCE_BYTES):
        """Start a new download budget; ``download`` raises ``TooLarge`` when it is used up."""
        self.resource_limit = limit
        self.resource_bytes = 0

    def check_public(self, url: str):
        """Refuse anything but http(s) URLs on public hosts, so scraped links cannot reach internal services."""
        parsed = urlsplit(url)
        host = parsed.hostname
        if parsed.scheme not in ('http', 'https') or not host:
            raise UnsafeUrl(f'{url!r}: not an http(s) URL')
        if time.time() - self._public_hosts.get(host, -_HOST_TTL) < _HOST_TTL:
            return
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise UnsafeUrl(f'{url!r}: IP literal hosts are refused')
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
        except socket.gaierror as err:
            raise ResourceTransient(f'cannot resolve {host!r}: {err!r}') from err
        if not addresses or not all(ipaddress.ip_address(item).is_global for item in addresses):
            raise UnsafeUrl(f'{url!r}: {host!r} resolves to a non-public address')
        self._public_hosts[host] = time.time()

    def _pace(self, host: str):
        wait = self._last.get(host, 0.0) + self.gaps.get(host, self.gap) - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def _send(self, method: str, url: str, follow: bool, kwargs) -> requests.Response:
        """One request, following redirects by hand so that every hop is checked and paced."""
        for _ in range(_MAX_REDIRECTS + 1):
            self.check_public(url)
            self._pace(urlsplit(url).hostname or '')
            resp = self.session.request(method, url, allow_redirects=False, **kwargs)
            if not (follow and resp.status_code in _REDIRECT_STATUS and resp.headers.get('Location')):
                return resp
            url = urljoin(resp.url, resp.headers['Location'])
            resp.close()
            if resp.status_code == 303 or (resp.status_code in (301, 302) and method == 'POST'):
                method = 'GET'
                kwargs = {key: value for key, value in kwargs.items() if key not in ('data', 'json')}
        raise UnexpectedResponse(310, '', 'too many redirects', url)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault('timeout', self.timeout)
        follow = kwargs.pop('allow_redirects', True)
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._send(method, url, follow, kwargs)
            except (requests.ConnectionError, requests.Timeout) as err:
                if attempt == self.retries:
                    raise ResourceTransient(f'{method} {url!r} failed: {err!r}') from err
                logging.warning(f'{method} {url!r} failed ({err!r}), retry {attempt}/{self.retries} ...')
                time.sleep(2 ** attempt)
                continue

            if resp.status_code in _RETRY_STATUS and attempt < self.retries:
                delay = _retry_after(resp) or 2 ** attempt
                resp.close()
                logging.warning(f'{method} {url!r} -> {resp.status_code}, retry in {delay:.0f}s ...')
                time.sleep(min(delay, 30.0))
                continue
            return resp
        raise AssertionError('unreachable')  # pragma: no cover

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request('GET', url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request('POST', url, **kwargs)

    def head(self, url: str, **kwargs) -> requests.Response:
        return self.request('HEAD', url, **kwargs)

    def download(self, url: str, dest: str, *, reject_html: bool = True, **kwargs) -> DownloadInfo:
        """Stream ``url`` into ``dest``; raise instead of leaving a partial or HTML file behind."""
        os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
        resp = self.request('GET', url, stream=True, **kwargs)
        try:
            content_type = resp.headers.get('Content-Type', '')
            if resp.status_code != 200:
                raise UnexpectedResponse(resp.status_code, content_type, _peek(resp), url)
            if reject_html and content_type.lower().startswith('text/html'):
                raise UnexpectedResponse(resp.status_code, content_type, _peek(resp), url)

            announced = resp.headers.get('Content-Length')
            if self.resource_limit and announced and self.resource_bytes + int(announced) > self.resource_limit:
                raise TooLarge(f'{url!r} announces {announced} bytes, over the resource budget')

            written = 0
            try:
                with open(dest, 'wb') as f:
                    for chunk in resp.iter_content(_CHUNK_SIZE):
                        written += len(chunk)
                        if self.resource_limit and self.resource_bytes + written > self.resource_limit:
                            raise TooLarge(f'{url!r} went over the resource budget')
                        f.write(chunk)
            except TooLarge:
                _remove(dest)
                raise
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as err:
                _remove(dest)
                raise ResourceTransient(f'Stream of {url!r} broke: {err!r}') from err

            size = os.path.getsize(dest)
            expected = resp.headers.get('Content-Length')
            if expected and not resp.headers.get('Content-Encoding') and int(expected) != size:
                _remove(dest)
                raise ResourceTransient(f'{url!r} truncated: {size} of {expected} bytes')
            if size == 0:
                _remove(dest)
                raise ResourceTransient(f'{url!r} returned an empty body')
            self.resource_bytes += size
            return DownloadInfo(size, content_type, dict(resp.headers), resp.url)
        finally:
            resp.close()


def _retry_after(resp: requests.Response) -> Optional[float]:
    value = resp.headers.get('Retry-After', '')
    return float(value) if value.isdigit() and float(value) > 0 else None


def _peek(resp: requests.Response) -> str:
    chunk = next(resp.iter_content(4096), b'')
    return chunk.decode('utf-8', errors='replace')


def _remove(path: str):
    if os.path.exists(path):
        os.remove(path)
